# `train_joint.py`

## 功能

执行 Stage 4 联合微调。每个 cycle 包含低光 teacher retention 和去雾训练。
`--experiment` 自动选择同实验的 Stage 3 initializer、低光 teacher、低光保持数据
和输出目录。

| 实验 | teacher | 低光保持数据 | initializer | 输出 |
|---|---|---|---|---|
| `lolv1` | `weights/lolv1.pth` | `dataset/LOLv1/Train/input` | `runs/lolv1/.../stage3_real/latest.pth` | `runs/lolv1/.../stage4_joint` |
| `lolv2real` | `weights/lolv2real.pth` | `dataset/LOLv2/Real_captured/Train/Low` | `runs/lolv2real/.../stage3_real/latest.pth` | `runs/lolv2real/.../stage4_joint` |
| `sice` | `weights/sice.pth` | `dataset/SICE/Train` | `runs/sice/.../stage3_real/latest.pth` | `runs/sice/.../stage4_joint` |

## 参数

- `--experiment {lolv1,lolv2real,sice}`：必填；
- `--config configs/joint.yaml`：必填；
- `--resume`：精确恢复 Stage 4；
- `--max-iterations`、`--device`：运行覆盖；
- `--teacher-checkpoint`、`--lowlight-data`、`--init-checkpoint`、`--output`：
  消融路径覆盖。

```bash
python train_joint.py \
  --experiment lolv2real \
  --config configs/joint.yaml \
  --max-iterations 100000
```

默认 lowlight:dehaze 更新比为 `1:1`，去雾 real:indoor 采样概率为
`0.7:0.3`。进度包含完整 cycle 耗时、平均速度和 ETA。
