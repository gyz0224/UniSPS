# UniSPS 训练与评测流程

## 1. 三组实验

`--experiment` 同时决定低光初始权重、Stage 4 低光保持数据、原生低光测试集、
Stage 1–4 checkpoint、推理结果和指标目录。

| 命令参数 | 初始权重 | Stage 4 低光数据 | 原生低光测试集 | 产物目录 |
|---|---|---|---|---|
| `--experiment lolv1` | `weights/lolv1.pth` | LOL-v1 Train/input | LOL-v1 Test | `runs/lolv1/` |
| `--experiment lolv2real` | `weights/lolv2real.pth` | LOL-v2 Real Train/Low | LOL-v2 Real Test | `runs/lolv2real/` |
| `--experiment sice` | `weights/sice.pth` | SICE Train | SICE Test | `runs/sice/` |

下文每组命令都先用注释标出 `--experiment`，随后给出可直接执行的完整命令。
`--experiment` 在实际命令中仍必须写在 Python 脚本名之后。

## 2. Stage 1–4 训练

| Stage | 作用 | 数据 | 初始化 | 训练量 | 输出目录 |
|---|---|---|---|---:|---|
| 1 | 让去雾分支适应低光 SPS 特征 | ITS | 当前实验的低光权重 | 10,000 iter | `stage1_warmup` |
| 2 | 训练合成室内去雾 | ITS | Stage 1 | 150,000 iter | `stage2_its` |
| 3 | 适配真实室外雾和天空区域 | RESIDE-unpaired + mask | Stage 2 | 100,000 iter | `stage3_real` |
| 4 | 保持低光能力并联合微调去雾 | 当前实验的原生低光数据 + ITS + 真实雾 | Stage 3 + 低光 teacher | 100,000 iter | `stage4_joint` |

### Stage 1

```bash
# --experiment lolv1
python train_dehaze.py \
  --experiment lolv1 \
  --config configs/dehaze_its.yaml \
  --stage 1 \
  --max-iterations 10000

# --experiment lolv2real
python train_dehaze.py \
  --experiment lolv2real \
  --config configs/dehaze_its.yaml \
  --stage 1 \
  --max-iterations 10000

# --experiment sice
python train_dehaze.py \
  --experiment sice \
  --config configs/dehaze_its.yaml \
  --stage 1 \
  --max-iterations 10000
```

输出分别位于：

```text
runs/lolv1/checkpoints/stage1_warmup/latest.pth
runs/lolv2real/checkpoints/stage1_warmup/latest.pth
runs/sice/checkpoints/stage1_warmup/latest.pth
```

### Stage 2

Stage 2 会自动读取同一实验的 Stage 1。

```bash
# --experiment lolv1
python train_dehaze.py \
  --experiment lolv1 \
  --config configs/dehaze_its.yaml \
  --stage 2 \
  --max-iterations 150000

# --experiment lolv2real
python train_dehaze.py \
  --experiment lolv2real \
  --config configs/dehaze_its.yaml \
  --stage 2 \
  --max-iterations 150000

# --experiment sice
python train_dehaze.py \
  --experiment sice \
  --config configs/dehaze_its.yaml \
  --stage 2 \
  --max-iterations 150000
```

输出分别位于：

```text
runs/lolv1/checkpoints/stage2_its/latest.pth
runs/lolv2real/checkpoints/stage2_its/latest.pth
runs/sice/checkpoints/stage2_its/latest.pth
```

### Stage 3

先检查真实数据和 mask：

```bash
python scripts/validate_real_dataset.py
```

Stage 3 会自动读取同一实验的 Stage 2。

```bash
# --experiment lolv1
python train_dehaze.py \
  --experiment lolv1 \
  --config configs/dehaze_real.yaml \
  --stage 3 \
  --max-iterations 100000

# --experiment lolv2real
python train_dehaze.py \
  --experiment lolv2real \
  --config configs/dehaze_real.yaml \
  --stage 3 \
  --max-iterations 100000

# --experiment sice
python train_dehaze.py \
  --experiment sice \
  --config configs/dehaze_real.yaml \
  --stage 3 \
  --max-iterations 100000
```

输出分别位于：

```text
runs/lolv1/checkpoints/stage3_real/latest.pth
runs/lolv2real/checkpoints/stage3_real/latest.pth
runs/sice/checkpoints/stage3_real/latest.pth
```

### Stage 4

Stage 4 自动读取同一实验的 Stage 3，选择相同实验的低光权重作为 teacher，
并使用第 1 节表格中的原生低光训练数据做能力保持。

```bash
# --experiment lolv1
python train_joint.py \
  --experiment lolv1 \
  --config configs/joint.yaml \
  --max-iterations 100000

# --experiment lolv2real
python train_joint.py \
  --experiment lolv2real \
  --config configs/joint.yaml \
  --max-iterations 100000

# --experiment sice
python train_joint.py \
  --experiment sice \
  --config configs/joint.yaml \
  --max-iterations 100000
```

输出分别位于：

```text
runs/lolv1/checkpoints/stage4_joint/latest.pth
runs/lolv2real/checkpoints/stage4_joint/latest.pth
runs/sice/checkpoints/stage4_joint/latest.pth
```

## 3. 评测预设怎么工作

同一个去雾预设需要执行两条命令：

1. `eval_dehaze.py` 加载对应 checkpoint，生成去雾图片；
2. `measure_dehaze.py` 读取图片，计算该数据集支持的指标。

结果和指标会按实验自动隔离：

```text
runs/<experiment>/results/dehaze/<stage>/<dataset>/
runs/<experiment>/metrics/dehaze/<预设名>.json
```

## 4. 原生低光测试集：低光能力

`pretrained` 测初始低光权重，`stage4` 测联合训练后的最终权重。两者都在
当前实验的原生测试集上计算 PSNR、SSIM 和 LPIPS。

| experiment | 输入 | GT | 样本与配对 |
|---|---|---|---|
| `lolv1` | `dataset/LOLv1/Test/input` | `dataset/LOLv1/Test/target` | 15 对同名图片 |
| `lolv2real` | `dataset/LOLv2/Real_captured/Test/Low` | `dataset/LOLv2/Real_captured/Test/Normal` | 100 对同名图片 |
| `sice` | `dataset/SICE/Test/image` | `dataset/SICE/Test/label` | 150 张曝光图共享 50 张场景 GT，例如 `10_1.JPG -> 10.JPG` |

### `pretrained`

```bash
# --experiment lolv1
python eval_lowlight.py --experiment lolv1 --stage pretrained
python measure.py --experiment lolv1 --stage pretrained --device cuda:0

# --experiment lolv2real
python eval_lowlight.py --experiment lolv2real --stage pretrained
python measure.py --experiment lolv2real --stage pretrained --device cuda:0

# --experiment sice
python eval_lowlight.py --experiment sice --stage pretrained
python measure.py --experiment sice --stage pretrained --device cuda:0
```

### `stage4`

```bash
# --experiment lolv1
python eval_lowlight.py --experiment lolv1 --stage stage4
python measure.py --experiment lolv1 --stage stage4 --device cuda:0

# --experiment lolv2real
python eval_lowlight.py --experiment lolv2real --stage stage4
python measure.py --experiment lolv2real --stage stage4 --device cuda:0

# --experiment sice
python eval_lowlight.py --experiment sice --stage stage4
python measure.py --experiment sice --stage stage4 --device cuda:0
```

默认输出：

```text
runs/lolv1/results/lowlight/{pretrained,stage4}/lolv1_test/
runs/lolv2real/results/lowlight/{pretrained,stage4}/lolv2real_test/
runs/sice/results/lowlight/{pretrained,stage4}/sice_test/
runs/<experiment>/metrics/lowlight/{pretrained,stage4}.json
```

三个实验使用的测试集不同，因此绝对指标不能直接横向比较；应在同一 experiment
内比较 `pretrained` 与 `stage4`。此前误用 LOL-v1 得到的 LOLv2-real 结果保留在
`cross_lolv1_test`，不会再写入原生指标文件。

## 5. SOTS Indoor：合成室内去雾

SOTS Indoor 有 clean GT，计算 PSNR、SSIM 和 CIEDE2000。

| 预设 | checkpoint | 配置 | 结果目录 |
|---|---|---|---|
| `stage2` | `stage2_its/latest.pth` | `dehaze_its.yaml` | `stage2/sots_indoor` |
| `stage3` | `stage3_real/latest.pth` | `dehaze_real.yaml` | `stage3/sots_indoor` |
| `stage4` | `stage4_joint/latest.pth` | `dehaze_its.yaml` | `stage4/sots_indoor` |

```bash
# --experiment lolv1
python eval_dehaze.py --experiment lolv1 --stage stage2
python measure_dehaze.py --experiment lolv1 --stage stage2
python eval_dehaze.py --experiment lolv1 --stage stage3
python measure_dehaze.py --experiment lolv1 --stage stage3
python eval_dehaze.py --experiment lolv1 --stage stage4
python measure_dehaze.py --experiment lolv1 --stage stage4

# --experiment lolv2real
python eval_dehaze.py --experiment lolv2real --stage stage2
python measure_dehaze.py --experiment lolv2real --stage stage2
python eval_dehaze.py --experiment lolv2real --stage stage3
python measure_dehaze.py --experiment lolv2real --stage stage3
python eval_dehaze.py --experiment lolv2real --stage stage4
python measure_dehaze.py --experiment lolv2real --stage stage4

# --experiment sice
python eval_dehaze.py --experiment sice --stage stage2
python measure_dehaze.py --experiment sice --stage stage2
python eval_dehaze.py --experiment sice --stage stage3
python measure_dehaze.py --experiment sice --stage stage3
python eval_dehaze.py --experiment sice --stage stage4
python measure_dehaze.py --experiment sice --stage stage4
```

## 6. SOTS Outdoor：合成室外去雾

SOTS Outdoor 有 clean GT，计算 PSNR、SSIM 和 CIEDE2000。三个预设统一使用
室外大气光配置。

| 预设 | checkpoint | 结果目录 |
|---|---|---|
| `stage2-outdoor` | `stage2_its/latest.pth` | `stage2/sots_outdoor` |
| `stage3-outdoor` | `stage3_real/latest.pth` | `stage3/sots_outdoor` |
| `stage4-outdoor` | `stage4_joint/latest.pth` | `stage4/sots_outdoor` |

```bash
# --experiment lolv1
python eval_dehaze.py --experiment lolv1 --stage stage2-outdoor
python measure_dehaze.py --experiment lolv1 --stage stage2-outdoor
python eval_dehaze.py --experiment lolv1 --stage stage3-outdoor
python measure_dehaze.py --experiment lolv1 --stage stage3-outdoor
python eval_dehaze.py --experiment lolv1 --stage stage4-outdoor
python measure_dehaze.py --experiment lolv1 --stage stage4-outdoor

# --experiment lolv2real
python eval_dehaze.py --experiment lolv2real --stage stage2-outdoor
python measure_dehaze.py --experiment lolv2real --stage stage2-outdoor
python eval_dehaze.py --experiment lolv2real --stage stage3-outdoor
python measure_dehaze.py --experiment lolv2real --stage stage3-outdoor
python eval_dehaze.py --experiment lolv2real --stage stage4-outdoor
python measure_dehaze.py --experiment lolv2real --stage stage4-outdoor

# --experiment sice
python eval_dehaze.py --experiment sice --stage stage2-outdoor
python measure_dehaze.py --experiment sice --stage stage2-outdoor
python eval_dehaze.py --experiment sice --stage stage3-outdoor
python measure_dehaze.py --experiment sice --stage stage3-outdoor
python eval_dehaze.py --experiment sice --stage stage4-outdoor
python measure_dehaze.py --experiment sice --stage stage4-outdoor
```

## 7. HSTS Synthetic：合成室外去雾

HSTS Synthetic 有 10 对 GT，计算 PSNR、SSIM、CIEDE2000 和 LPIPS。

| 预设 | checkpoint | 结果目录 |
|---|---|---|
| `stage2-hsts-synthetic` | `stage2_its/latest.pth` | `stage2/hsts_synthetic` |
| `stage3-hsts-synthetic` | `stage3_real/latest.pth` | `stage3/hsts_synthetic` |
| `stage4-hsts-synthetic` | `stage4_joint/latest.pth` | `stage4/hsts_synthetic` |

```bash
# --experiment lolv1
python eval_dehaze.py --experiment lolv1 --stage stage2-hsts-synthetic
python measure_dehaze.py --experiment lolv1 --stage stage2-hsts-synthetic
python eval_dehaze.py --experiment lolv1 --stage stage3-hsts-synthetic
python measure_dehaze.py --experiment lolv1 --stage stage3-hsts-synthetic
python eval_dehaze.py --experiment lolv1 --stage stage4-hsts-synthetic
python measure_dehaze.py --experiment lolv1 --stage stage4-hsts-synthetic

# --experiment lolv2real
python eval_dehaze.py --experiment lolv2real --stage stage2-hsts-synthetic
python measure_dehaze.py --experiment lolv2real --stage stage2-hsts-synthetic
python eval_dehaze.py --experiment lolv2real --stage stage3-hsts-synthetic
python measure_dehaze.py --experiment lolv2real --stage stage3-hsts-synthetic
python eval_dehaze.py --experiment lolv2real --stage stage4-hsts-synthetic
python measure_dehaze.py --experiment lolv2real --stage stage4-hsts-synthetic

# --experiment sice
python eval_dehaze.py --experiment sice --stage stage2-hsts-synthetic
python measure_dehaze.py --experiment sice --stage stage2-hsts-synthetic
python eval_dehaze.py --experiment sice --stage stage3-hsts-synthetic
python measure_dehaze.py --experiment sice --stage stage3-hsts-synthetic
python eval_dehaze.py --experiment sice --stage stage4-hsts-synthetic
python measure_dehaze.py --experiment sice --stage stage4-hsts-synthetic
```

## 8. HSTS Real-world：真实室外去雾

HSTS Real-world 有 10 张真实雾图，没有 clean GT，只计算 FADE。三个预设默认
使用 `tile-size=1024`、`tile-overlap=128`。

| 预设 | checkpoint | 结果目录 |
|---|---|---|
| `stage2-hsts-real` | `stage2_its/latest.pth` | `stage2/hsts_real` |
| `stage3-hsts-real` | `stage3_real/latest.pth` | `stage3/hsts_real` |
| `stage4-hsts-real` | `stage4_joint/latest.pth` | `stage4/hsts_real` |

```bash
# --experiment lolv1
python eval_dehaze.py --experiment lolv1 --stage stage2-hsts-real
python measure_dehaze.py --experiment lolv1 --stage stage2-hsts-real --workers 4
python eval_dehaze.py --experiment lolv1 --stage stage3-hsts-real
python measure_dehaze.py --experiment lolv1 --stage stage3-hsts-real --workers 4
python eval_dehaze.py --experiment lolv1 --stage stage4-hsts-real
python measure_dehaze.py --experiment lolv1 --stage stage4-hsts-real --workers 4

# --experiment lolv2real
python eval_dehaze.py --experiment lolv2real --stage stage2-hsts-real
python measure_dehaze.py --experiment lolv2real --stage stage2-hsts-real --workers 4
python eval_dehaze.py --experiment lolv2real --stage stage3-hsts-real
python measure_dehaze.py --experiment lolv2real --stage stage3-hsts-real --workers 4
python eval_dehaze.py --experiment lolv2real --stage stage4-hsts-real
python measure_dehaze.py --experiment lolv2real --stage stage4-hsts-real --workers 4

# --experiment sice
python eval_dehaze.py --experiment sice --stage stage2-hsts-real
python measure_dehaze.py --experiment sice --stage stage2-hsts-real --workers 4
python eval_dehaze.py --experiment sice --stage stage3-hsts-real
python measure_dehaze.py --experiment sice --stage stage3-hsts-real --workers 4
python eval_dehaze.py --experiment sice --stage stage4-hsts-real
python measure_dehaze.py --experiment sice --stage stage4-hsts-real --workers 4
```

## 9. I-HAZE：真实室内去雾

I-HAZE 有 30 对真实雾图和 clean GT，计算 PSNR、SSIM 和 CIEDE2000。三个预设
默认使用 `tile-size=1024`、`tile-overlap=128`。

| 预设 | checkpoint | 配置 | 结果目录 |
|---|---|---|---|
| `stage2-ihaze` | `stage2_its/latest.pth` | `dehaze_its.yaml` | `stage2/ihaze` |
| `stage3-ihaze` | `stage3_real/latest.pth` | `dehaze_real.yaml` | `stage3/ihaze` |
| `stage4-ihaze` | `stage4_joint/latest.pth` | `dehaze_real.yaml` | `stage4/ihaze` |

```bash
# --experiment lolv1
python eval_dehaze.py --experiment lolv1 --stage stage2-ihaze
python measure_dehaze.py --experiment lolv1 --stage stage2-ihaze
python eval_dehaze.py --experiment lolv1 --stage stage3-ihaze
python measure_dehaze.py --experiment lolv1 --stage stage3-ihaze
python eval_dehaze.py --experiment lolv1 --stage stage4-ihaze
python measure_dehaze.py --experiment lolv1 --stage stage4-ihaze

# --experiment lolv2real
python eval_dehaze.py --experiment lolv2real --stage stage2-ihaze
python measure_dehaze.py --experiment lolv2real --stage stage2-ihaze
python eval_dehaze.py --experiment lolv2real --stage stage3-ihaze
python measure_dehaze.py --experiment lolv2real --stage stage3-ihaze
python eval_dehaze.py --experiment lolv2real --stage stage4-ihaze
python measure_dehaze.py --experiment lolv2real --stage stage4-ihaze

# --experiment sice
python eval_dehaze.py --experiment sice --stage stage2-ihaze
python measure_dehaze.py --experiment sice --stage stage2-ihaze
python eval_dehaze.py --experiment sice --stage stage3-ihaze
python measure_dehaze.py --experiment sice --stage stage3-ihaze
python eval_dehaze.py --experiment sice --stage stage4-ihaze
python measure_dehaze.py --experiment sice --stage stage4-ihaze
```

## 10. RTTS：真实室外去雾

RTTS 有 4,322 张真实雾图，没有 clean GT，只计算 FADE。当前比较 Stage 3 和
Stage 4。

| 预设 | checkpoint | 结果目录 |
|---|---|---|
| `stage3-real` | `stage3_real/latest.pth` | `stage3/rtts` |
| `stage4-real` | `stage4_joint/latest.pth` | `stage4/rtts` |

```bash
# --experiment lolv1
python eval_dehaze.py --experiment lolv1 --stage stage3-real
python measure_dehaze.py --experiment lolv1 --stage stage3-real --workers 4
python eval_dehaze.py --experiment lolv1 --stage stage4-real
python measure_dehaze.py --experiment lolv1 --stage stage4-real --workers 4

# --experiment lolv2real
python eval_dehaze.py --experiment lolv2real --stage stage3-real
python measure_dehaze.py --experiment lolv2real --stage stage3-real --workers 4
python eval_dehaze.py --experiment lolv2real --stage stage4-real
python measure_dehaze.py --experiment lolv2real --stage stage4-real --workers 4

# --experiment sice
python eval_dehaze.py --experiment sice --stage stage3-real
python measure_dehaze.py --experiment sice --stage stage3-real --workers 4
python eval_dehaze.py --experiment sice --stage stage4-real
python measure_dehaze.py --experiment sice --stage stage4-real --workers 4
```

## 11. 断点继续

去雾推理中断后，直接重新运行相同的 `eval_dehaze.py` 命令。脚本会跳过已有且
尺寸正确的结果。

训练从 checkpoint 继续时，每个实验使用自己的 `--resume` 路径：

```bash
# --experiment lolv1
python train_dehaze.py \
  --experiment lolv1 \
  --config configs/dehaze_its.yaml \
  --stage 2 \
  --resume runs/lolv1/checkpoints/stage2_its/iter_0050000.pth

# --experiment lolv2real
python train_dehaze.py \
  --experiment lolv2real \
  --config configs/joint.yaml \
  --stage 4 \
  --resume runs/lolv2real/checkpoints/stage4_joint/iter_0066000.pth

# --experiment sice
python train_dehaze.py \
  --experiment sice \
  --config configs/dehaze_its.yaml \
  --stage 2 \
  --resume runs/sice/checkpoints/stage2_its/iter_0050000.pth
```
