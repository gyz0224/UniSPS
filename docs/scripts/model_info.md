# `tools/model_info.py`

## 功能

构造完整 SPS 多任务模型并统计总参数量和 `requires_grad=True` 的参数量。脚本会自动把项目根目录加入 `sys.path`，因此可直接从 `tools/` 路径执行。

## 参数

- `--device`：模型初始化设备；默认 CUDA 可用时用 CUDA，否则 CPU。

该命令会构造默认 `SemanticPriorNet`，因此需要可用的 CLIP ViT-B/16 缓存或下载条件。

```bash
python tools/model_info.py --device cpu
```

代码层也可复用：

```python
from tools.model_info import count_params

total = count_params(model, trainable_only=False)
trainable = count_params(model, trainable_only=True)
```
