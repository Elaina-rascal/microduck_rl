"""一键可视化：python train/visual_flat.py。"""

import os
from pathlib import Path

TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
NUM_ENVS = 10
CHECKPOINT_FILE = None  # 指定模型文件；None 表示自动选择。
RUN_DIR = None  # 指定训练目录；None 表示最新的、已有模型的正式训练。
CHECKPOINT_ITERATION = None  # 例如 250；None 表示最新轮数。


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    if CHECKPOINT_FILE:
        checkpoint = Path(CHECKPOINT_FILE).expanduser()
    else:
        models = list(Path(RUN_DIR).expanduser().glob("model_*.pt")) if RUN_DIR else list(
            Path("logs/rsl_rl/velocity").glob("*/model_*.pt")
        )
        if not models:
            raise FileNotFoundError("没有找到模型，请先训练或设置 CHECKPOINT_FILE。")
        run_dir = max(p.parent for p in models)
        checkpoint = (
            run_dir / f"model_{CHECKPOINT_ITERATION}.pt"
            if CHECKPOINT_ITERATION is not None
            else max(
                (p for p in models if p.parent == run_dir),
                key=lambda p: int(p.stem.split("_")[1]),
            )
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"模型不存在：{checkpoint}")
    print(f"加载模型：{checkpoint}", flush=True)

    from mjlab.scripts.play import PlayConfig, run_play

    run_play(TASK_ID, PlayConfig(
        checkpoint_file=str(checkpoint), num_envs=NUM_ENVS, viewer="viser",
    ))


if __name__ == "__main__":
    main()
