"""`train` console script — deliberately identical to mjlab's own.

This project must keep DECLARING a `train` script even though it no longer
needs one: `--hf-jobs` is intercepted from the `mjlab.tasks` plugin entry point
(train_hook.py), on mjlab's own import path, so the wrapper has nothing left to
do.

Why keep it anyway: same-name console scripts are last-writer-wins, and after
an install BOTH dist-info RECORDs claim `.venv/bin/train`. Removing this
declaration therefore does not hand the name back to mjlab — it makes the next
`uv sync` UNINSTALL the file (ours, per our RECORD) while nothing reinstalls
mjlab's. `train` then vanishes from the venv entirely and `uv run train` falls
through to whatever `train` sits on PATH: on one machine liblinear's, which
answered `can't open input file Mjlab-Velocity-Flat-MicroDuck` (2026-08-31).

So the collision stays, and is made harmless instead: whichever shim wins,
`train` behaves the same, and the flag is handled in exactly one place.
"""

from __future__ import annotations

# 中文阅读起点：激活 .venv 后，最小训练命令为：
# train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5
# 本文件负责转交给 mjlab；安装后的 train 也可能直接指向 mjlab.scripts.train.main，效果相同。
# 按顺序阅读：tasks/__init__.py 的任务注册 → microduck_velocity_env_cfg.py 的任务配方
# → mdp.py 的具体计算函数 → robot/microduck_constants.py 的机器人与执行器配置。

import sys


def main() -> int | None:
    # 外部接口 mjlab.scripts.train.main()：无显式参数，从 sys.argv 读取任务 ID 和配置覆盖项。
    # --env.* 修改环境配置；--agent.* 修改网络/PPO/运行配置。实际解析由 mjlab + tyro 完成。
    # 导入 mjlab 时，它先按 pyproject.toml 的 mjlab.tasks 入口加载本仓库的任务注册模块。
    # This import runs mjlab's plugin loader, which imports
    # mjlab_microduck.tasks -> train_hook.maybe_submit_to_hf_jobs(). A
    # `--hf-jobs` invocation submits and exits inside the import below; it
    # never comes back here.
    from mjlab.scripts.train import main as mjlab_train_main

    # mjlab 内部主线（调用处说明，外部库源码不修改）：
    # 1. 任务注册表 → env_cfg、agent_cfg、runner_cls，随后用命令行参数覆盖默认值。
    # 2. ManagerBasedRlEnv(cfg=env_cfg, device=...) → N 个并行环境。
    #    reset() → (obs_dict, extras)；step(actions[N,14]) → (obs_dict, reward[N],
    #    terminated[N], truncated[N], extras)。走路的 obs_dict 含 actor[N,61]、critic[N,76]。
    # 3. RslRlVecEnvWrapper → rsl_rl 的 VecEnv 接口，观测包装为 TensorDict；合并两类结束标记，
    #    另在 extras["time_outs"] 保留超时信息，供 PPO 做价值补偿。
    # 4. runner_cls(env, asdict(agent_cfg), log_dir, device) → 含 Actor/Critic/PPO 的 Runner。
    #    runner.learn(num_learning_iterations=..., init_at_random_ep_len=True) 无模型返回值，
    #    按“每环境采样 24 步 → GAE → PPO 更新”循环，模型和配置保存到 logs/rsl_rl/velocity/。
    # 本调用同步运行直到训练结束；返回值仅作为命令退出码使用，不是 policy 或 checkpoint。
    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())
