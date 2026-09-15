# `measure_dehaze.py`

## 功能

测量一个实验的去雾结果，并保存 JSON：

- paired 数据（SOTS Indoor/Outdoor、HSTS Synthetic、I-HAZE）：PSNR、SSIM、CIEDE2000 和 AlexNet LPIPS；
- HSTS Real-world 和 RTTS：无参考 FADE。

```bash
python measure_dehaze.py \
  --experiment lolv2real \
  --stage stage4-hsts-synthetic

python measure_dehaze.py \
  --experiment lolv2real \
  --stage stage4-real \
  --workers 4
```

默认读取
`runs/<experiment>/results/dehaze/<stage>/<dataset>/`，JSON 写入
`runs/<experiment>/metrics/dehaze/<preset>.json`。

## Stage 预设

| 数据集 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|
| SOTS Indoor | `stage2` | `stage3` | `stage4` |
| SOTS Outdoor | `stage2-outdoor` | `stage3-outdoor` | `stage4-outdoor` |
| HSTS Synthetic | `stage2-hsts-synthetic` | `stage3-hsts-synthetic` | `stage4-hsts-synthetic` |
| HSTS Real-world | `stage2-hsts-real` | `stage3-hsts-real` | `stage4-hsts-real` |
| I-HAZE | `stage2-ihaze` | `stage3-ihaze` | `stage4-ihaze` |
| RTTS | — | `stage3-real` | `stage4-real` |

## 参数

- `--experiment`、`--stage`：必填；
- `--prediction`、`--reference`、`--pairing`：覆盖预设；
- `--metrics-output`：覆盖 JSON 路径；
- `--lpips-device`：paired 数据集的 LPIPS 设备；
- `--workers`：仅 FADE 的图片级 CPU 并行；
- `--print-every`：进度间隔。

预测和 GT 不会自动缩放，尺寸不一致会报错。FADE 越低表示感知残雾越少，但仍需
结合视觉质量和下游检测，避免把过度增强误判为更好。
