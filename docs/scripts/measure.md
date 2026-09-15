# `measure.py`

## 功能

测量有监督低光结果的 PSNR、SSIM、AlexNet LPIPS 和 NIQE，并把结果保存为实验专属 JSON。

```bash
python measure.py \
  --experiment sice \
  --stage pretrained \
  --device cuda:0

python measure.py \
  --experiment sice \
  --stage stage4 \
  --device cuda:0
```

默认预测和 GT 都由 `--experiment` 选择：

| experiment | 预测目录 | GT | 配对 |
|---|---|---|---|
| `lolv1` | `lolv1_test/I` | `dataset/LOLv1/Test/target` | 同名 |
| `lolv2real` | `lolv2real_test/I` | `dataset/LOLv2/Real_captured/Test/Normal` | 同名 |
| `sice` | `sice_test/I` | `dataset/SICE/Test/label` | `10_1.JPG -> 10.JPG` |

JSON 写入：

```text
runs/sice/metrics/lowlight/pretrained.json
runs/sice/metrics/lowlight/stage4.json
```

`--im_dir`、`--label_dir`、`--pairing {same-name,sice}` 和
`--metrics-output` 可覆盖默认值。PSNR、SSIM、LPIPS 会在预测缩放到 GT 尺寸后计算；
无参考 NIQE 直接使用预测结果的原始输出尺寸计算，且越低越好。
图片 glob 同时支持 PNG/JPG/JPEG/BMP，输入为空或 GT 缺失会立即报错。
