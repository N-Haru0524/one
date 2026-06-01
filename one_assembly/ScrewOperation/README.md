# ScrewOperation

ねじ取りつけ補正用 ViT のデータ収集・学習・評価パイプライン。実機 (KHIBunri + ROS2) と
Isaac sim の両系統で **同じ samples.csv スキーマ** に出力するので、学習以降のスクリプトは
データ出処を区別せず使える。

詳細仕様:
- 設定スキーマ → [docs/screw_config.md](docs/screw_config.md)
- ディレクトリ構成 → [docs/dataset_layout.md](docs/dataset_layout.md)

---

## 全体フロー

```
[gen_pose_csv.py]    ── poses.csv ───┐
   host (one)                        ▼
                            [Isaac dataset.py]  ── images / samples.csv
                            container (Isaac)

  もしくは

[data_collector.py]  ────────────────── images / samples.csv
   実機 (host + ROS2 bridge)


[train_vit_spiral.py]   samples.csv + images  →  model.pt
[infer_vit_spiral.py]   1 ペア画像  →  predicted class + dx, dy
[eval_vit_spiral_hist.py]  samples.csv 全体  →  位置誤差ヒスト + 散布図
```

---

## Sim 経路: gen_pose_csv → Isaac dataset.py

### 1. 姿勢 CSV を生成 (host, ~10 秒)

```bash
cd /home/wrs/nagai/one
uv run python -m one_assembly.ScrewOperation.gen_pose_csv \
    --session rly_scrw_pick \
    --worklist_root one_assembly/worklists/electric_assembly \
    --prescrew_offset 0.002 \
    --rgt_ee_extension max \
    --num_classes 91 \
    --spiral_step 0.0008 \
    --sequence rly_scrw --mode pick \
    --description "rly_scrw pickup #1, prescrew 2mm, shank max"
```

主なオプション:
- `--session <token>_pick[:<history>]` — WorkList から prescrew TCP を計算 (`rly_scrw_pick`, `blt_scrw_pick`, …)
  - history で完了済みねじを指定して 2 本目以降を狙う: `rly_scrw_pick:rly_scrw`
- `--prescrew <yaml>` — 既存の prescrew YAML を直接読む (`--session` と排他)
- `--prescrew_offset` — bit 退避距離 [m] (screw 軸方向)
- `--rgt_ee_extension` — shank 伸長 [m]、`max` で完全伸長 (= 0.033164)
- `--flip_axis` — screw 軸の向きを反転 (layout 規約が逆のとき)
- `--num_classes` / `--spiral_step` — spiral 形状 (WRS デフォルト 91 / 0.0008m)
- `--out <path>` — 明示的出力 (デフォルトは `datasets/{stage}/{NNN}/poses.csv`)
- `--preview` — pyglet で 91 IK 解を順次再生 (X11 + `./dev_gui.sh` 必要)

出力:
```
datasets/train/001/
├── poses.csv             # ↓ Isaac の入力
├── poses.csv.gen.yaml    # prescrew + spiral 設定 + IK 失敗クラス一覧
└── config.yaml           # ScrewConfig snapshot
```

IK 失敗チェック (空であること):
```bash
grep ik_failed datasets/train/001/poses.csv.gen.yaml
```

### 2. Isaac sim で画像 + samples.csv を生成 (~40 秒、ストリーミング一時停止)

```bash
cd /home/wrs/nagai/Isaac_sim
./scripts/isaac.sh run /workspace/scripts/dataset.py \
    --ep_dir /workspace_screw/datasets/train/001
```

`/workspace_screw` は `one_assembly/ScrewOperation/` の rw マウント (isaac.sh が設定)。
gen_pose_csv 終了時に表示される `Next step:` ヒントをそのままコピペすれば OK。

ep_dir の中身が以下のようになる:
```
datasets/train/001/
├── poses.csv             # 入力 (上で生成)
├── poses.csv.gen.yaml
├── config.yaml
├── samples.csv           ← Isaac が追加
├── meta.jsonl            ← Isaac が追加 (T_world_link6 + T_world_cam)
└── images/               ← Isaac が追加
    ├── 000000_cam1.png
    ├── 000000_cam2.png
    └── ...               # 91 × 2 = 182 枚
```

コンテナ内で `os.chown` するので host から見ても `wrs:wrs` 所有。

### 3. 完了確認

```bash
EP=one_assembly/ScrewOperation/datasets/train/001
ls $EP/images/ | wc -l        # → 182
wc -l $EP/samples.csv          # → 92
```

### 4. ROI を目視確認 (推奨)

ViT は `roi1` / `roi2` で各カメラ画像を crop する。デフォルト `(0, 0, 320, 240)` は 640×480
画像の左上 1/4 を切り出すので、bit が画面外なら学習が成立しない。任意のフレームに ROI を
重ねた overlay を出して確認:

```bash
EP=one_assembly/ScrewOperation/datasets/train/001
uv run python -c "
from PIL import Image, ImageDraw
import yaml, os
cfg = yaml.safe_load(open('$EP/config.yaml'))
out = '/tmp/roi_check'
os.makedirs(out, exist_ok=True)
for idx in (0, 45, 90):
    for cam, roi in (('cam1', tuple(cfg['roi1'])), ('cam2', tuple(cfg['roi2']))):
        im = Image.open(f'$EP/images/{idx:06d}_{cam}.png').convert('RGB')
        d = ImageDraw.Draw(im)
        d.rectangle([roi[0], roi[1], roi[2]-1, roi[3]-1], outline='red', width=4)
        d.text((10, 10), f'idx={idx} {cam} ROI={roi} img={im.size}', fill='red')
        im.save(f'{out}/{idx:06d}_{cam}_overlay.png')
print(f'wrote overlays under {out}/')
"
xdg-open /tmp/roi_check/000000_cam1_overlay.png
```

ねじ + bit が赤枠の中に入っているか目視。外れているなら ROI を変更して再学習
([ROI を変えるとき](#roi-を変えるとき) 参照)。

---

## 実機経路: data_collector.py

```bash
uv run python -m one_assembly.ScrewOperation.data_collector \
    --session rly_scrw_pick \
    --worklist_root one_assembly/worklists/electric_assembly \
    --prescrew_offset 0.002 \
    --rgt_ee 0.0 \
    --num_classes 91 \
    --sequence rly_scrw --mode pick
```

事前に ROS2 + `one_planner_bridge` + 実機 (KHI + OnRobot SD) + USB カメラが立ち上がっていること。
出力先は同じ `datasets/train/{NNN}/`。

---

## 学習

```bash
uv run python -m one_assembly.ScrewOperation.train_vit_spiral \
    --train_dirs datasets/train/001
# → datasets/model/{NNN}/ に model.pt + config.yaml を自動採番で作成
```

主なオプション:
- `--train_dirs A B C` — 複数 ep を結合して学習可能 (相対パスは `ScrewOperation/` 起点)
- `--val_dir <path>` — 別 ep を val に使う (省略時は train を流用 → overfit デバッグ用)
- `--epochs` / `--batch_size` / `--lr` / `--roi1` / `--roi2` / `--num_classes` — config.yaml の値を override

WRS 準拠の **デフォルト値で回すなら CLI から override 系を渡さない**:
- epochs=50, batch_size=256, lr=3e-4, num_classes=91, patch_size=5, dim/depth/heads/k=128/12/8/64,
  resize_per_cam=(45,40), spiral_step=0.0008

---

## 推論 / 評価

### 単発推論 (1 ペア画像)

```bash
uv run python -m one_assembly.ScrewOperation.infer_vit_spiral \
    --model_dir datasets/model/001 \
    --cam1 datasets/train/001/images/000000_cam1.png \
    --cam2 datasets/train/001/images/000000_cam2.png
# → predicted class: 0, correction dx, dy [m]: 0.0 0.0
```

### 全 frame 評価 + 位置誤差ヒストグラム

```bash
EP_INFER=datasets/infer/001
mkdir -p $EP_INFER

uv run python -m one_assembly.ScrewOperation.eval_vit_spiral_hist \
    --csv datasets/train/001/samples.csv \
    --image_dir datasets/train/001/images \
    --model datasets/model/001/model.pt \
    --config datasets/model/001/config.yaml \
    --num_accu 0.0008 \
    --out_dir $EP_INFER \
    --success_mm 0.5 1.0 2.0
```

要注意:
- `--config` を明示的に渡す (script の自動検出は `*_config.yaml` を探すので名前不一致)
- `--num_accu` は学習時の `spiral_step` と一致させる (デフォルト 0.002 はズレるので注意)

出力:
- `eval_results.csv` — 全 frame の予測クラス + 位置誤差 [mm] + 成功フラグ
- `hist_pos_err_mm.png` — 位置誤差ヒスト
- `scatter_with_vectors.png` — XY 散布図 + 補正ベクトル

論文用にスタイルを変えた図が欲しいときは `fig_vit_spiral_hist.py` (引数同じ)。

---

## マルチねじ収集

スクリュラックの 2 本目以降は session の history で counter を進める。`make_mode_dir` が
自動で NNN を採番するので、毎回 `datasets/train/{NNN}/` が増えていく。

```bash
# 1本目
uv run python -m one_assembly.ScrewOperation.gen_pose_csv \
    --session rly_scrw_pick ...                       # → datasets/train/001/

# 2本目 (1本目完了状態)
uv run python -m one_assembly.ScrewOperation.gen_pose_csv \
    --session "rly_scrw_pick:rly_scrw" ...            # → datasets/train/002/

# 3本目
uv run python -m one_assembly.ScrewOperation.gen_pose_csv \
    --session "rly_scrw_pick:rly_scrw-rly_scrw" ...   # → datasets/train/003/
```

ラック何本分まで取れるかは layout `screw.pitch` と物理空間の制約による
([layouts.yaml](../worklists/electric_assembly/yamls/layouts.yaml))。

複数 ep 結合学習:
```bash
uv run python -m one_assembly.ScrewOperation.train_vit_spiral \
    --train_dirs datasets/train/001 datasets/train/002 datasets/train/003 \
    --val_dir datasets/train/004
```

---

## ROI を変えるとき

ROI は **画像 crop なので画像取り直し不要、再学習のみで反映される**。

1. `config.py` の `roi1` / `roi2` デフォルトを編集 (新規 ep 全部に効く)、または既存ep の
   `config.yaml` を直接編集
2. `train_vit_spiral.py` 起動時の CLI でその場 override:
   ```bash
   uv run python -m one_assembly.ScrewOperation.train_vit_spiral \
       --train_dirs datasets/train/001 \
       --roi1 160 120 480 360 \
       --roi2 160 120 480 360
   ```
   ※ PIL 規約: `(left, upper, right, lower)`。`(160,120,480,360)` は 320×240 を画像中央から切る。

`resize_per_cam` (現状 45×40) もアスペクト比に合わせて再検討すること。ViT 入力サイズは
`(45, 40 × 2 cam) = (45, 80)` で patch_size=5 なので 9×16 = 144 patch。

---

## トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| frame 0 だけ画像が初期姿勢 | PD 整定不足。`--first-settle-steps 60` (default) を増やす |
| frame 0 がブレ画像 | DLSS temporal blend。`--first-render-steps 25` を増やす |
| bit が screw 真上ではない | shank が IK に反映されてない (古いコード)。`--rgt_ee_extension max` を明示 |
| 物体が描画されてない (preview) | `_build_worklist` の `pos` オフセットが効いてない (古いコード) |
| `ik_failed_classes` に多数 | prescrew_offset を変える / `--flip_axis` を試す / WorkList layout を確認 |
| `samples.csv` / `images/` が root 所有 | isaac.sh に `HOST_UID/GID` env が無い (旧版)。最新版に更新 |
| ViT が学習されない (val_acc 上がらない) | ROI が画像内のねじを外している可能性大。上記 [ROI 目視確認](#4-roi-を目視確認-推奨) |
| `unrecognized arguments: --description` | argparse から漏れ。`gen_pose_csv.py` を更新 |

---

## このディレクトリの構成

```
ScrewOperation/
├── README.md                      # ← 今読んでるファイル
├── docs/
│   ├── screw_config.md            # ScrewConfig フィールド早見
│   └── dataset_layout.md          # ディレクトリ構成
├── config.py                      # ScrewConfig (pydantic)
├── config/
│   ├── cameras.yaml               # 実機カメラ設定
│   └── prescrew.example.yaml      # prescrew YAML のサンプル
├── gen_pose_csv.py                # sim 用 poses.csv 生成
├── data_collector.py              # 実機収集
├── camera.py                      # DualCameraRecorder
├── correction_loop.py             # 補正ループ state machine
├── screw_correction_run.py        # 推論ループ (sim/実機共通)
├── bridge_io.py                   # ROS2 bridge ラッパ
├── prescrew.py                    # prescrew TCP / IK 解決
├── session.py                     # ScrewSession (タスク識別)
├── spiral_metry.py                # hex_ring_abs (spiral 座標)
├── utils.py                       # make_mode_dir, rot6d, csv_writer, …
├── approach_plan.py               # ScrewPlanner draft → SyncPlan 変換
├── capture_prescrew.py            # /right/joint_states スナップショット
├── dataset.py                     # SpiralDataset (学習用)
├── model_builder.py               # build_vit
├── train_vit_spiral.py            # 学習
├── infer_vit_spiral.py            # 単発推論
├── eval_vit_spiral_hist.py        # 全 frame 評価 + ヒスト
├── fig_vit_spiral_hist.py         # 論文用図
└── make_dummy_spiral.py           # ダミーデータ生成 (smoke 用)
```

Isaac sim 側の対応物: `/home/wrs/nagai/Isaac_sim/scripts/dataset.py`,
`/home/wrs/nagai/Isaac_sim/scripts/isaac.sh`。
