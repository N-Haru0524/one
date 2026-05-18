# データセット dir 構成

`one_assembly/ScrewOperation/datasets/` 配下のレイアウト。実機 (`data_collector`) と
sim (`gen_pose_csv` + Isaac `dataset.py`) で共通。stage は `train` / `infer` / `model`
の 3 種、ep 番号 NNN は自動採番 (`utils.make_mode_dir`)。

```
datasets/
├── train/
│   └── NNN/
│       ├── config.yaml             # ScrewConfig snapshot
│       ├── samples.csv             # 学習データ本体
│       ├── images/
│       │   ├── 000000_cam1.png
│       │   ├── 000000_cam2.png
│       │   └── ...
│       ├── poses.csv               # sim のみ: gen_pose_csv の入力 CSV
│       ├── poses.csv.gen.yaml      # sim のみ: prescrew + spiral 設定
│       └── meta.jsonl              # sim のみ: T_world_link6 / T_world_cam
│
├── model/
│   └── NNN/
│       ├── config.yaml             # ScrewConfig snapshot
│       └── model.pt                # 学習済みモデル
│
└── infer/
    └── NNN/
        ├── config.yaml
        ├── samples.csv             # 推論で集めた画像と pose
        ├── images/
        ├── eval_results.csv        # eval_vit_spiral_hist の出力
        ├── hist_pos_err_mm.png     # 位置誤差ヒストグラム
        └── scatter_with_vectors.png  # 補正ベクトル散布図
```

## sequence / mode はパスではなく config.yaml に入れる

過去には `datasets/{sequence}/{mode}/{stage}/NNN/` の入れ子レイアウトを使っていたが、
現在は **フラット** (`datasets/{stage}/NNN/`)。タスク識別子 (`sequence`) と動作 phase
(`mode`) は `config.yaml` の同名フィールドにだけ書き込む。
([screw_config.md](screw_config.md))

```bash
# gen_pose_csv / data_collector ともに --sequence/--mode 指定可だが、
# パスには反映されない (config.yaml だけに乗る)
uv run python -m one_assembly.ScrewOperation.gen_pose_csv \
    --session rly_scrw_pick \
    --sequence rly_scrw --mode pick \
    ...
# → datasets/train/001/ に着地
# → config.yaml に sequence: rly_scrw / mode: pick が記録される
```

## samples.csv のスキーマ

実機 collector と Isaac sim で **共通列 + 観測 DOF 列** の構成。

| 列 | 由来 | 内容 |
|---|---|---|
| `idx` | 共通 | フレーム番号 (0..) |
| `time` | 共通 | タイムスタンプ |
| `label` | meta passthrough | spiral クラス id |
| `dx`, `dy` | meta passthrough | TCP 絶対オフセット [m] |
| `tcp_x`, `tcp_y`, `tcp_z` | meta passthrough | コマンド時の TCP 位置 [m] |
| `r6d_0`..`r6d_5` | meta passthrough | TCP 姿勢 (Zhou et al. 6D 表現) |
| `right_joint1..6` | sim のみ | 観測 arm DOF [rad] |
| `right_sd_shank_joint` | sim のみ | 観測 shank 伸び幅 [m] (0..0.033164) |
| `left_*` | sim のみ | 左腕 DOF (現状未制御、stage 既定値) |

実機 collector は meta 列のみを書く (DOF は ROS の `/right/joint_states` 経由でとれる)。
sim 側は両方書く: meta は入力 `poses.csv` から透過コピー、DOF は articulation の `get_joint_positions()`。

## 画像命名

両ソースとも `{idx:06d}_cam1.png` / `{idx:06d}_cam2.png`。`cam1/cam2` はカメラ番号
(yaml キーの `cam0`/`cam1` とはオフセット 1 ずれる歴史的経緯あり; [config/cameras.yaml](../config/cameras.yaml) 参照)。

## Isaac sim から書き出すときのマウント

Isaac コンテナ (`/workspace/`) から ScrewOperation サブツリーは `/workspace_screw/` として
rw マウントされる ([Isaac_sim/scripts/isaac.sh](../../../../Isaac_sim/scripts/isaac.sh) `SRC_MOUNTS`)。
ep_dir はその下の絶対パスを渡す:

```bash
./scripts/isaac.sh run /workspace/scripts/dataset.py \
    --ep_dir /workspace_screw/datasets/train/001
```

コンテナが root で動くので、Isaac が書き出した `samples.csv` / `images/*.png` /
`meta.jsonl` は **root 所有**。書き換えたければ
`sudo chown -R wrs:wrs datasets/` で chown する。
