# 平地行走训练

在项目根目录激活已安装依赖的虚拟环境，然后直接运行：

```bash
source .venv/bin/activate
python train/train_flat.py
```

无需命令行参数。脚本先做 64 个环境、5 次迭代的检查，通过后从头开始
4096 个环境、6000 次迭代的正式训练，需要可用的 CUDA GPU。

参数集中在 `train_flat.py` 顶部。只运行检查时设置 `SMOKE_TEST_ONLY = True`。
当前 `LOGGER = "wandb"`，正式训练需要登录 W&B；只需本地日志时改为
`"tensorboard"`。启动前的检查始终使用 TensorBoard。

正式训练的模型、配置与日志位于 `logs/rsl_rl/velocity/`，检查日志位于
`logs/rsl_rl/velocity_smoke/`。导出部署模型时使用项目现有的
`scripts/export.py`，以便将观测归一化器烘焙进 ONNX。

## 路径怎么理解

- **当前工作目录（cwd）**：程序读写相对文件路径的起点，可用 `Path.cwd()` 查看。
  从终端启动时通常就是终端当前所在目录。
- **脚本所在目录**：`train_flat.py` 所在的 `train/`，通过
  `Path(__file__).resolve().parent` 获取，和你从哪里启动无关。
- **Python 解释器路径**：决定使用哪个 Python 和其中安装的依赖，与上述两个目录
  是不同的概念。激活 `.venv` 后，`python` 通常指向项目的 `.venv/bin/python`。
- **模块搜索路径（`sys.path`）**：`import` 用它寻找模块，与相对文件读写的起点不同。
  切换 cwd 不会自动把项目的 `src/` 安装为可导入的包。

例如，从项目根目录运行 `python train/train_flat.py`，启动时 cwd 是项目根目录，
脚本目录是项目下的 `train/`。从 `train/` 运行 `python train_flat.py`，启动时
cwd 则是 `train/`。终端中命令里的相对脚本路径，也要从当时的 cwd 开始寻找。

本脚本先用 `__file__` 找到项目根目录，再执行 `os.chdir(PROJECT_ROOT)`。
因此这两种启动方式最终都会把日志写到项目根目录下的 `logs/`。
代码中的路径注释逐项解释了 `resolve()`、`parents[1]` 和 `chdir()`。
