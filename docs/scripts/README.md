# 可执行脚本功能索引

本目录逐一说明 SPS-Netpro 的所有可执行 Python 入口。所有入口统一采用：

```python
build_parser()
parse_args(argv=None)
main(argv=None) -> int

if __name__ == "__main__":
    raise SystemExit(main())
```

因此脚本可被测试或其他 Python 代码安全导入，不会在 import 时解析参数、加载模型、创建数据集或启动训练。

Stage 1–4、低光/去雾推理和指标入口必须传
`--experiment {lolv1,lolv2real,sice}`。生成产物统一位于
`runs/<experiment>/`；完整命令见
[训练与评测流程](../TRAINING_PIPELINE.md)。

## 低光脚本

| 脚本 | 角色 | 文档 |
|---|---|---|
| `train_lowlight.py` | 配对参考指标低光训练主入口 | [train_lowlight.md](train_lowlight.md) |
| `train_lowlight_unpaired.py` | NIQE/BRISQUE/MUSIQ 低光训练入口 | [train_lowlight_unpaired.md](train_lowlight_unpaired.md) |
| `eval_lowlight.py` | 低光结果与中间特征导出入口 | [eval_lowlight.md](eval_lowlight.md) |
| `measure.py` | PSNR/SSIM/LPIPS 目录评估 | [measure.md](measure.md) |

## 去雾与工具脚本

| 脚本 | 角色 | 文档 |
|---|---|---|
| `train_dehaze.py` | Stage 1–3 无配对去雾训练 | [train_dehaze.md](train_dehaze.md) |
| `train_joint.py` | Stage 4 低光/去雾联合训练 | [train_joint.md](train_joint.md) |
| `eval_dehaze.py` | 支持原分辨率重叠分块的部署去雾推理 | [eval_dehaze.md](eval_dehaze.md) |
| `measure_dehaze.py` | Stage 2/3/4 的 SOTS、HSTS、I-HAZE paired/LPIPS/FADE 与 RTTS FADE | [measure_dehaze.md](measure_dehaze.md) |
| `scripts/flist.py` | 递归生成确定性图像列表 | [flist.md](flist.md) |
| `scripts/validate_real_dataset.py` | 校验真实室外图像与天空 mask | [validate_real_dataset.md](validate_real_dataset.md) |
| `tools/model_info.py` | 模型参数量统计 | [model_info.md](model_info.md) |

源码模块职责和兼容层关系见 [../SOURCE_MODULES.md](../SOURCE_MODULES.md)。
