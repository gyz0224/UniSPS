# `eval_lowlight.py`

## 功能

评测实验的低光初始权重或 Stage 4 联合权重，并导出增强图与中间特征。

| 参数 | 作用 |
|---|---|
| `--experiment` | 必填：`lolv1`、`lolv2real` 或 `sice` |
| `--stage` | `pretrained` 或 `stage4`，默认 `pretrained` |
| `--data_test` | 覆盖当前 experiment 的原生测试输入 |
| `--model` | 覆盖自动 checkpoint |
| `--output-folder` | 覆盖自动结果目录 |
| `--sanitize-zeros` | 将精确为 0 的输入通道替换为 `1/255`；仅用于曾出现黑斑的目标域数据 |
| `--threads`、`--device` | DataLoader worker 和设备 |

| experiment | 默认输入 | 默认结果数据集目录 |
|---|---|---|
| `lolv1` | `dataset/LOLv1/Test/input` | `lolv1_test` |
| `lolv2real` | `dataset/LOLv2/Real_captured/Test/Low` | `lolv2real_test` |
| `sice` | `dataset/SICE/Test/image` | `sice_test` |

```bash
python eval_lowlight.py \
  --experiment sice \
  --stage pretrained

python eval_lowlight.py \
  --experiment sice \
  --stage stage4
```

LOLv1、LOLv2 和 SICE 论文基准默认保留原始零值，不传
`--sanitize-zeros`，以复现原始表格。对 VisDrone、DroneVehicle 等曾出现
Retinex 黑斑的目标域数据显式开启：

```bash
python eval_lowlight.py \
  --experiment lolv1 \
  --stage stage4 \
  --data_test /path/to/target/images \
  --output-folder /path/to/enhanced/output \
  --sanitize-zeros
```

默认输出分别位于：

```text
runs/sice/results/lowlight/pretrained/sice_test/
runs/sice/results/lowlight/stage4/sice_test/
```

每个目录含 `I/L/R/X`、子图、照明、噪声反射、中间特征和语义特征。输入会先按
开关决定是否替换零值，再裁成偶数尺寸；最长边超过 1024 时等比例缩放并对齐到
8 的倍数。
