# UniSPS 源码模块功能参考

本文档描述重组后的每个维护中 Python 模块。可执行命令的逐项参数和输出另见 [scripts/README.md](scripts/README.md)。

## 1. 组织原则

```text
CLI / main guard
        │
        ▼
train_*.py / eval_*.py
        │
        ├── training/  训练步骤、运行时、采样、checkpoint
        ├── datasets/  数据集与 loader factory
        ├── loss/      低光和去雾损失
        ├── metrics/   PSNR/SSIM/LPIPS/CIEDE2000/FADE
        └── net/       blocks → task subnet → top-level model
```

旧重复模块已经删除，规范实现只存在一份。模型顶层属性
`L_net/R_net/N_net/sem_net/illp/Gamma_Predictor/dehaze_head` 没有改名，
因此 state-dict key 不变。

## 2. 网络模块 `net/`

| 文件 | 功能与主要接口 |
|---|---|
| `net/blocks.py` | 共享 `LayerNorm`、`Attention`、`IGA`、`SGA`、FFN、`TransformerBlock`、`IGABlock`、`OverlapPatchEmbed`；低光和去雾共同依赖。 |
| `net/pafm.py` | `PAFM` 的规范实现；融合两个同形状特征的跨通道和空间注意。 |
| `net/semantic.py` | `SemanticPriorNet`；冻结 ViT-B/16，`forward()` 返回 RGB/768ch 空间语义，`encode_global()` 返回可向输入回传的全局 embedding。 |
| `net/lowlight.py` | checkpoint 兼容的 `L_net`、`R_net`、`N_net`、`Illumination_Estimator`、`Gamma_Predictor`。 |
| `net/model.py` | 顶层 `net`/`SPSNet`；构造顺序和属性名保持不变，路由 `lowlight`/`dehaze`，提供物理范围切换和部署 wrapper。 |
| `net/dehaze.py` | `DehazeConfig`、max/DCP 大气光、物理逆解、三尺度 SPS 去雾参数头、部署模型。 |
| `net/depth.py` | 训练专用轻量相对深度网络，输出限定在配置范围。 |
| `net/rehaze.py` | 训练专用物理加雾器和幅度不超过 0.1 的残差细化器。 |
| `net/discriminator.py` | 清晰域/雾域 PatchGAN/LSGAN 判别器构造。 |
| `net/__init__.py` | 包级导出 `net` 和 `SPSNet`。 |

## 3. 数据模块

| 文件 | 功能与主要接口 |
|---|---|
| `datasets/lowlight.py` | 原 `DatasetFromFolder` 与 `DatasetFromFolderEval`；保留场景目录/平面目录行为。 |
| `datasets/dehaze.py` | `UnpairedDehazeDataset`；D4+ 独立 clean/hazy/reference 随机采样、mask 同步变换、测试 GT 边界；集中实现真实图/mask basename 与尺寸校验。 |
| `datasets/loaders.py` | 原 256 随机裁剪训练 transform、ToTensor 评估 transform，以及 `get_training_set/get_eval_set`。 |
| `datasets/__init__.py` | 数据包公共导出。 |

## 4. 损失模块 `loss/`

| 文件 | 功能与主要接口 |
|---|---|
| `loss/lowlight_loss.py` | `C_loss/R_loss/P_loss`、曝光、空间、TV、颜色损失；公式不变，tensor 改为跟随输入设备。 |
| `loss/sasw_loss.py` | `WHTBlock`、`CosineLoss`、`SASWLoss`。 |
| `loss/lowlight_clip.py` | 冻结 ViT-B/32 的低光 semantic consistency 与 prompt-IQA 目标。 |
| `loss/dehaze_loss.py` | cycle、LSGAN、beta/depth 伪监督、冻结 VGG19 双重对比感知、复用 CLIP 语义一致性及总损失。 |
| `loss/__init__.py` | 统一导出全部低光/去雾损失。 |

## 5. 训练模块 `training/`

| 文件 | 功能与主要接口 |
|---|---|
| `training/lowlight_sampling.py` | Neighboring Pixel Masking、2×2 subimage、pair downsampling、gamma augmentation；generator 跟随输入设备。 |
| `training/lowlight_trainer.py` | `LowlightTrainer`、低光损失配置、seed/device、paired/no-reference 验证、checkpoint 过滤/加载、epoch 范围解析。 |
| `training/dehaze_trainer.py` | Stage freeze、互斥 optimizer、D→G→Depth 三段更新、finite diagnostics、日志统计、完整 checkpoint。 |
| `training/experiment.py` | `lolv1/lolv2real/sice` 实验身份、原生低光数据与配对规则、Stage 1–4 checkpoint 链、去雾预设、结果/指标目录和防覆盖检查。 |
| `training/runtime.py` | YAML、device、无配对 loader、训练栈、scheduler 和日志格式化。 |
| `training/__init__.py` | 统一导出训练器、采样、checkpoint 和 stage 工具。 |

## 6. 指标和通用工具

| 文件 | 功能与主要接口 |
|---|---|
| `metrics/image_quality.py` | 单一来源的 `ssim/calculate_ssim/calculate_psnr`。 |
| `metrics/lpips_evaluator.py` | 共享 AlexNet LPIPS 模型、同尺寸 RGB 图对计算，以及增强图目录 PSNR/SSIM/LPIPS 平均。 |
| `metrics/fade.py` | LIVE FADE 1.0 的本地 Python 移植；内嵌官方 500/500 图像训练的参考模型，提供单张 8-bit RGB 图像的无参考雾密度分数和可选密度图。 |
| `metrics/dehaze_evaluator.py` | SOTS/HSTS/I-HAZE 严格配对，PSNR/SSIM/CIEDE2000、HSTS 可选 LPIPS，以及无 GT 目录的 FADE 平均。 |
| `metrics/__init__.py` | 指标包公共导出。 |
| `image_utils.py` | PIL RGB/灰度图水平拼接。 |

## 7. 可执行入口

| 文件 | 功能 | 详细文档 |
|---|---|---|
| `train_lowlight.py` | 配对参考低光训练 | [scripts/train_lowlight.md](scripts/train_lowlight.md) |
| `train_lowlight_unpaired.py` | 无参考 IQA 低光训练 | [scripts/train_lowlight_unpaired.md](scripts/train_lowlight_unpaired.md) |
| `eval_lowlight.py` | 低光输出/特征评估 | [scripts/eval_lowlight.md](scripts/eval_lowlight.md) |
| `measure.py` | PSNR/SSIM/LPIPS 测量 | [scripts/measure.md](scripts/measure.md) |
| `measure_dehaze.py` | SOTS、HSTS、I-HAZE 去雾全参考/LPIPS 与 HSTS/RTTS FADE 测量 | [scripts/measure_dehaze.md](scripts/measure_dehaze.md) |
| `train_dehaze.py` | Stage 1–3 去雾训练 | [scripts/train_dehaze.md](scripts/train_dehaze.md) |
| `train_joint.py` | Stage 4 联合训练 | [scripts/train_joint.md](scripts/train_joint.md) |
| `eval_dehaze.py` | 部署去雾推理、断点续跑及原分辨率重叠分块 | [scripts/eval_dehaze.md](scripts/eval_dehaze.md) |
| `scripts/flist.py` | 图像 flist 生成 | [scripts/flist.md](scripts/flist.md) |
| `scripts/validate_real_dataset.py` | D4+ 真实数据与天空 mask 全量校验 | [scripts/validate_real_dataset.md](scripts/validate_real_dataset.md) |
| `tools/model_info.py` | 参数量统计 | [scripts/model_info.md](scripts/model_info.md) |

## 8. 测试模块

| 文件 | 覆盖内容 |
|---|---|
| `tests/helpers.py` | 下载无关语义 prior、tiny model、随机 batch。 |
| `tests/test_lowlight_backward_compat.py` | 默认/显式低光一致及 legacy state keys。 |
| `tests/test_organization.py` | package 公共导出、规范入口 import safety、实验路径隔离、Stage 4 teacher、CLI 预设、重叠分块融合、CPU sampling/loss、低光完整更新。 |
| `tests/test_dehaze_metrics.py` | CIEDE2000、LPIPS 归一化、FADE 回归、数据集配对、尺寸检查和 paired/FADE 分流。 |
| `tests/test_dehaze_forward.py` | 去雾 shape/range/finite、DCP/max、部署边界、受限 refiner。 |
| `tests/test_unpaired_dataset.py` | D4+ 独立随机采样、mask、GT 污染拒绝、真实数据布局校验。 |
| `tests/test_optimizer_param_groups.py` | stage freeze 与 optimizer 参数互斥。 |
| `tests/test_training_smoke.py` | 两次三段更新和完整 checkpoint 往返。 |
| `tests/__init__.py` | 测试 package 标记。 |

## 9. 规范导入速查

| 功能 | 规范路径 |
|---|---|
| 顶层模型 | `net.model.net` 或 `from net import net` |
| 语义先验 | `net.semantic.SemanticPriorNet` |
| 低光子网 | `net.lowlight` |
| 共享注意力模块 | `net.blocks` |
| PAFM | `net.pafm.PAFM` |
| 低光/去雾数据集 | `datasets.lowlight` / `datasets.dehaze` |
| loader factory | `datasets.loaders` |
| 低光损失 | `loss.lowlight_loss` / `loss.sasw_loss` |
| 低光采样 | `training.lowlight_sampling` |
| CLIP 低光损失 | `loss.lowlight_clip.CLIPLoss` |

项目代码只使用以上规范路径；旧根模块和大小写兼容路径不再提供。生成产物统一
使用 `runs/<experiment>/`，不再使用根目录 `checkpoints/` 和 `results/`。
