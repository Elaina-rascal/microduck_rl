"""平地行走训练：激活项目 .venv 后执行 python train/train_flat.py。"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path


# 修改这里即可，无需传命令行参数。
TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
NUM_ENVS = 4096
MAX_ITERATIONS = 6000
SEED = 42
GPU_IDS = [0]  # 使用第一张可见 GPU。
LOGGER = "wandb"  # W&B 记录，需要登录；改成 "tensorboard" 可只在本地记录。
RUN_NAME = "flat_walk"
SAVE_INTERVAL = 250

# 正式训练前自动检查；只想检查流程时，将 SMOKE_TEST_ONLY 改成 True。
SMOKE_TEST_ONLY = False
SMOKE_NUM_ENVS = 64
SMOKE_ITERATIONS = 5

# Python 中要区分两个目录：
# 1. 当前工作目录（cwd）：用 Path.cwd() 或 os.getcwd() 查看。
#    从终端启动时通常是终端所在目录，IDE 也可以单独配置它。
#    open("data.txt")、Path("logs") 等相对文件路径，默认从 cwd 开始找。
# 2. 脚本所在目录：本文件 train_flat.py 所在的 train/，不会因 cwd 改变。
#    Python 不会因为运行了某个脚本，就自动把 cwd 切换到脚本所在目录。
#
# __file__ 是当前 Python 文件的路径；Path 把它包装为路径对象。
# resolve() 得到绝对路径，并解析符号链接、消除路径中的 .. 等部分。
# 对 /home/elaina/project/microduck_rl/train/train_flat.py 来说：
#   .parent 或 .parents[0] -> /home/elaina/project/microduck_rl/train
#   .parents[1]           -> /home/elaina/project/microduck_rl（项目根目录）
# 这里从脚本自身定位项目，所以从哪个目录启动都能找到同一个项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_config(*, smoke: bool):
    from mjlab.scripts.train import TrainConfig

    cfg = TrainConfig.from_task(TASK_ID)
    cfg.env.scene.num_envs = SMOKE_NUM_ENVS if smoke else NUM_ENVS
    cfg.agent.max_iterations = SMOKE_ITERATIONS if smoke else MAX_ITERATIONS
    cfg.agent.seed = SEED
    cfg.agent.logger = "tensorboard" if smoke else LOGGER
    cfg.agent.run_name = f"{RUN_NAME}_smoke" if smoke else RUN_NAME
    cfg.agent.save_interval = SAVE_INTERVAL
    # 检查日志与正式训练分开；检查得到的权重不用于正式训练。
    if smoke:
        cfg.agent.experiment_name += "_smoke"
    return replace(cfg, gpu_ids=list(GPU_IDS))


def main() -> None:
    # 将当前进程的 cwd 切换到项目根目录，影响后续相对文件路径的解析。
    # 不会移动文件，也不会改变 __file__、Python 解释器或外部终端的目录。
    # mjlab 用相对路径 logs/rsl_rl/<experiment_name>/ 保存训练结果；
    # 切换后，正式训练日志会落到 PROJECT_ROOT / "logs/rsl_rl/velocity"，
    # 检查日志会落到 PROJECT_ROOT / "logs/rsl_rl/velocity_smoke"。
    # 注意：这影响的是文件读写路径；import 的查找路径由 sys.path 管理。
    # mjlab 和 mjlab_microduck 需要已安装在当前使用的 Python 环境中。
    os.chdir(PROJECT_ROOT)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到可用的 CUDA GPU。请确认使用项目 .venv 的 Python，"
            "并检查 NVIDIA 驱动和 PyTorch CUDA 环境。"
        )

    from mjlab.scripts.train import launch_training

    print("开始平地训练检查：64 个环境，5 次迭代。", flush=True)
    launch_training(TASK_ID, make_config(smoke=True))
    if SMOKE_TEST_ONLY:
        return

    print(
        f"检查通过，开始正式平地训练：{NUM_ENVS} 个环境，"
        f"{MAX_ITERATIONS} 次迭代。",
        flush=True,
    )
    launch_training(TASK_ID, make_config(smoke=False))


if __name__ == "__main__":
    main()
