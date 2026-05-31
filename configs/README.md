# Configs

MonoSpikeStereo uses YAML files for training, testing, and raw-spike inference.

| Folder | Entry |
| --- | --- |
| `train/` | Training and fine-tuning presets. |
| `test/` | Validation-style testing on NPZ files with stored `left`, `right`, and `mt`. |
| `inference/` | Raw-spike inference from `spk` and `LCD`, with optional saved inputs and videos. |

Local `*_my.yaml` files are ignored by git.
