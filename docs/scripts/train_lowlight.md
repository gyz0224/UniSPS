# `train_lowlight.py`

## 功能

使用原 SPS-Net Neighboring Pixel Masking 目标训练低光增强模型，并用配对参考图计算 PSNR、SSIM、LPIPS。该文件是配对参考低光训练的唯一入口。

## 主要参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--batchSize` | `1` | 训练 batch size |
| `--nEpochs` | `300` | 总 epoch |
| `--snapshots` | `1` | 配对验证间隔 |
| `--start_iter` | `1` | 起始 epoch；大于 1 时尝试加载前一 epoch |
| `--lr` | `1e-4` | Adam 学习率 |
| `--decay` / `--gamma` | `300` / `0.5` | MultiStepLR 周期和衰减倍数 |
| `--data_train` | `dataset/LOLv1/Train/input` | 485 张自监督低光训练图 |
| `--data_val` | `dataset/LOLv1/Test/input` | 15 张验证低光图 |
| `--reference_val` | `dataset/LOLv1/Test/target` | 15 张配对验证参考图 |
| `--loss_weights` | `1 0.1 0.1 0.5` | exposure/spatial/TV/color 权重 |
| `--light_patch` | `64` | 曝光损失池化窗口 |
| `--w_sem` / `--w_iqa` | `0.1` / `0.01` | CLIP 语义与质量项权重 |
| `--save_epochs` | 空 | 例如 `50,60-80` 的额外 checkpoint |
| `--device` | 空 | 显式覆盖为 `cpu`、`cuda:0` 等 |
| `--save_folder` | `weights/LOLv1` | 最佳模型和指定 epoch checkpoint |
| `--logroot` | `logs/LOLv1` | TensorBoard 日志 |

原 `--gpu_mode`、`--rgb_range` 参数继续保留。默认仍使用 GPU；CPU 运行需显式给出 `--device cpu`。

当前 `dataset/LOLv1/Train/target` 也包含 485 张配对正常光图，但自监督训练损失不会读取它；target 只在 `Test/target` 的验证指标中使用。

## 单步训练流程

`training.lowlight_trainer.LowlightTrainer` 执行：

1. 冻结语义先验前向，提取 ViT-B/16 语义；
2. 为每个 2×2 邻域生成两个互斥像素 mask；
3. 形成两张半分辨率子图，并对第二张执行随机 gamma；
4. 三次低光前向得到子图输出和完整图输出；
5. 计算一致性、重建、先验、曝光、空间、TV、颜色、SASW、CLIP semantic/IQA；
6. 按原权重求和并执行一次 Adam 更新。

## 验证与输出

- 验证参考图使用输入 basename；SICE 风格的 `_曝光编号` 后缀会被去掉；
- TensorBoard 记录 `psnr`、`ssim`、`lpips`；
- 保存 `best_psnr.pth`、`best_ssim.pth`、`best_lpips.pth`；
- `best_models_metrics.txt` 汇总各最佳模型的完整三项指标；
- `epoch_N.pth` 只在 `--save_epochs` 命中时保存；
- 所有模型文件排除冻结的 `sem_net.*` 权重。

## 示例

```bash
python train_lowlight.py \
  --data_train dataset/LOLv1/Train/input \
  --data_val dataset/LOLv1/Test/input \
  --reference_val dataset/LOLv1/Test/target \
  --save_folder weights/LOLv1 --logroot logs/LOLv1 \
  --lr 1e-5 --loss_weights 1 0.1 0.1 0.5 \
  --save_epochs 50,100,150,200,250,300
```
