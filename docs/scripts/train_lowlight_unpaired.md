# `train_lowlight_unpaired.py`

## 功能

面向 DICM、LIME、MEF、NPE、VV 等无配对/无参考场景训练低光增强模型。训练损失与 `train_lowlight.py` 共用同一个 `LowlightTrainer`，验证改用 NIQE、BRISQUE、MUSIQ。

## 关键参数

- `--data_train`：默认 `dataset/LOLv1/Train/input`；
- `--data_val`：默认 `dataset/LOLv1/Test/input`；
- `--save_folder`、`--logroot`：默认 `weights/LOLv1_Unpaired`、`logs/LOLv1_Unpaired`；
- `--weights_dir`：NIQE/BRISQUE/MUSIQ 本地权重目录；
- `--mean_val`：曝光目标，默认 `0.5`；
- 其余 batch、epoch、LR、损失参数与配对训练保持相同语义；
- `--device` 可显式选择设备。

兼容保留的 `--decay` 和 `--gamma` 与原脚本一样不创建 scheduler。

当前仓库没有 `weights/metrics_weights`，因此该入口不是默认 Stage 0 路径。运行前必须安装 `pyiqa` 并准备 NIQE、BRISQUE、MUSIQ 权重；本地可直接训练的规范流程是配对验证入口 `train_lowlight.py`。

## IQA 初始化

`build_no_reference_metrics()`：

1. 仅在真正运行训练时延迟导入 `pyiqa`；
2. 将本地 NIQE/BRISQUE/MUSIQ 权重同步到 PyTorch hub cache；
3. 创建 NIQE、BRISQUE；
4. 以 `pretrained=False` 创建 MUSIQ，再从 `weights_dir` 严格加载本地参数。

## 验证和 checkpoint

- 大图最长边缩放至 1024，并将尺寸向下对齐到 8 的倍数；
- NIQE、BRISQUE 越低越好，MUSIQ 越高越好；
- 保存 `best_niqe.pth`、`best_brisque.pth`、`best_musiq.pth`；
- 每个 epoch 保存 `epoch_N.pth`；
- checkpoint 继续排除 `sem_net.*`。

## 示例

```bash
python train_lowlight_unpaired.py \
  --data_train dataset/LOLv1/Train/input \
  --data_val dataset/LOLv1/Test/input \
  --save_folder weights/LOLv1_Unpaired \
  --logroot logs/LOLv1_Unpaired \
  --weights_dir weights/metrics_weights --lr 1e-4
```
