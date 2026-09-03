# `scripts/flist.py`

## 功能

递归扫描数据目录并生成确定性 `.flist`。支持 jpg/jpeg/png/bmp/tif/tiff，不区分扩展名大小写。

## 参数

- `--path`：必需的输入目录；
- `--output`：必需的输出文件；父目录自动创建；
- `--absolute`：写入 resolve 后的绝对路径；不指定时相对当前工作目录。

输入不存在、不是目录、没有受支持图像或输出无法写入时，会以 `flist error:` 给出明确错误。结果按 resolve 后路径排序，每行一个文件。

```bash
python scripts/flist.py --path dataset/ITS/hazy \
  --output dataset/flists/its_train_hazy.flist --absolute
```

当前 `configs/dehaze_its.yaml` 可直接读取 `dataset/ITS/gt` 和 `dataset/ITS/hazy`，无需先生成 flist；只有需要固定样本清单或跨目录组合时才使用本工具。

真实室外配置同样直接读取 `dataset/real/{clear,hazy}`；其图像与 mask 的一一对应关系使用 `scripts/validate_real_dataset.py` 检查，不由 flist 工具负责。
