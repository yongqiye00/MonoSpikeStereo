# Training Configs

Run:

```bash
python train.py --config configs/train/train.yaml
```

| Key | Meaning |
| --- | --- |
| `npz_path` | Directory or file containing training NPZ data. |
| `log_dir` | TensorBoard log directory. |
| `checkpoint_dir` | Directory for saved checkpoints. |
| `name` | Experiment name used in checkpoint filenames. |
| `net_type` | Model family selector. The public preset uses `flow`. |
| `device` | `cuda` or `cpu`. |
| `resume` | Continue optimizer/scheduler/epoch state from a training checkpoint. |
| `resume_path` | Explicit checkpoint path for `resume`. |
| `init_checkpoint` | Model weights used to initialize training without optimizer state. |
| `batch_size` | Batch size per optimizer step. |
| `num_workers` | DataLoader worker count. |
| `epochs` | Number of training epochs. |
| `save_every_epochs` | Periodic checkpoint save interval. |
| `max_train_samples` | Optional cap on training clips; `0` means no cap. |
| `max_val_batches` | Optional cap on validation batches; `0` means no cap. |
| `lr` | Learning rate. |
| `grad_clip_norm` | Gradient clipping norm; `0` disables clipping. |
| `sequence_length` | Number of frames per clip. |
| `train_clip_stride` | Start-frame stride for training clips. |
| `state_history` | Number of recurrent/history states kept by the model wrapper. |
| `crop_size` | Spatial crop size in `HxW` format. |
| `amp` | Enable CUDA automatic mixed precision. |
| `amp_dtype` | Autocast dtype, either `fp16` or `bf16`. `bf16` is more stable on supported GPUs. |
| `compile` | Enable `torch.compile` when available. |
