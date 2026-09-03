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

默认输出分别位于：

```text
runs/sice/results/lowlight/pretrained/sice_test/
runs/sice/results/lowlight/stage4/sice_test/
```

每个目录含 `I/L/R/X`、子图、照明、噪声反射、中间特征和语义特征。输入会先裁成
偶数尺寸；最长边超过 1024 时等比例缩放并对齐到 8 的倍数。
