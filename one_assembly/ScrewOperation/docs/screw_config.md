# ScrewConfig 早見

`one_assembly/ScrewOperation/config.py` で定義されている pydantic モデル。
データ収集 (`gen_pose_csv` / `data_collector`) と ViT 学習・推論 (`train_vit_spiral` /
`infer_vit_spiral` / `eval_vit_spiral_hist`) で共通使用される。

各 episode dir に `config.yaml` として保存され、`load_config(path)` で読み戻せる。

## スキーマ一覧

| field | 型 | デフォルト | 役割 | 主に書く側 |
|---|---|---|---|---|
| `description` | str | `""` | フリーテキストメモ | データ収集 (`--description`) |
| `sequence` | str | `""` | タスク識別子 (例: `rly_scrw`, `blt_scrw`) | データ収集 (`--sequence`) |
| `mode` | str | `""` | 動作 phase (例: `pick`, `place`) | データ収集 (`--mode`) |
| `roi1` | (x,y,w,h) | `(0,0,320,240)` | cam1 画像の crop 領域 | 学習 |
| `roi2` | (x,y,w,h) | `(0,0,320,240)` | cam2 画像の crop 領域 | 学習 |
| `num_classes` | int | `91` | spiral クラス数 (label の値域) | 収集 (`--num_classes`) |
| `spiral_step` | float | `0.0008` | spiral 半径ステップ [m] | 収集 (`--spiral_step`) |
| `patch_size` | int | `5` | ViT パッチサイズ | 学習 |
| `dim` | int | `128` | ViT 埋め込み次元 | 学習 |
| `depth` | int | `12` | ViT 層数 | 学習 |
| `heads` | int | `8` | ViT attention head 数 | 学習 |
| `k` | int | `64` | (用途) | 学習 |
| `channels` | int | `3` | 入力画像チャンネル数 | 学習 |
| `resize_per_cam` | (h,w) | `(45,40)` | ViT 入力リサイズ (per-cam) | 学習 |
| `epochs` | int | `50` | 学習 epoch 数 | 学習 |
| `batch_size` | int | `256` | 学習 batch | 学習 |
| `lr` | float | `3e-4` | 学習率 | 学習 |
| `num_workers` | int | `8` | DataLoader worker 数 | 学習 |
| `train_dirs` | list[str] | `[]` | 学習で読み込む ep_dir リスト | 学習 (`--train_dirs`) |
| `model_dir` | str | `""` | 学習モデルの出力先 | 学習 |
| `max_num_samples` | int | `91` | bridge 経路 1 ep の最大フレーム数 | 実機 collector |
| `latency` | float | `2.0` | bridge action 後の待機 [s] | 実機 collector |

派生プロパティ:
- `image_size` → `(h, w*2)` `resize_per_cam` から算出。cam1/cam2 を横連結した ViT 入力サイズ。

## 書込み・読込み

```python
from one_assembly.ScrewOperation.config import ScrewConfig, save_config, load_config

cfg = ScrewConfig(sequence="rly_scrw", mode="pick", num_classes=91)
save_config(cfg, "datasets/train/001/config.yaml")

cfg = load_config("datasets/train/001/config.yaml")
```

CLI からの上書きは `merge_cli_args(config, args)`。`ScrewConfig.model_fields` に存在する
field 名と一致する argparse オプションだけが反映される。

## 関連

- データセット dir 構成: [dataset_layout.md](dataset_layout.md)
