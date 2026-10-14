本文面向了解强化学习基本概念、第一次接触机器人 RL 工程的读者。以 **`Mjlab-Velocity-Flat-MicroDuck`（平地走路）** 为主线，说明算法如何连接仿真、训练和部署。

代码基线：本仓库 `b02fdea`；本机依赖 `mjlab 1.3.0`、`rsl-rl-lib 5.0.1`、`Warp 1.12.0`。核对日期：2026-10-13。其他任务共享基础设施，奖励、初始化和课程各不相同。

# 第一部分：核心算法实现

## 1. 三层代码分别负责什么

**本仓库定义“让机器人学什么、在什么物理条件下学”；PPO 的通用更新过程由依赖库实现。** 因此，仅在本仓库搜索 `PPO`，看不到完整训练算法。

| 层次 | 主要职责 | 从哪里读起 |
| --- | --- | --- |
| `microduck_rl` | 机器人、观测、动作、奖励、终止、随机化和课程 | `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` |
| `mjlab` | 管理环境、调用各项 MDP 函数、推进 MuJoCo Warp 仿真、对接训练器 | `mjlab/envs/manager_based_rl_env.py` |
| `rsl_rl` | Actor/Critic、采样缓存、GAE、PPO 损失和梯度更新 | `rsl_rl/algorithms/ppo.py` |

`mjlab`、`rsl_rl` 的路径位于本机 `.venv/lib/python3.12/site-packages/` 下。项目自己的路径均相对于仓库根目录。

运行入口是 `train_cli.py`，它转交给 `mjlab.scripts.train`。`tasks/__init__.py` 把任务 ID 与 **训练环境配置、播放环境配置、网络配置、Runner** 绑定。选任务就是选这一整套配方。

## 2. 策略的输入和输出

### 输入：Actor 的 61 维观测

Actor 只能看到部署时可以提供的信息。这里的“状态”是可观测量拼接出的向量，并非直接把完整仿真状态输入网络。

| 位置（从 0 开始） | 内容 | 维数 | 含义 |
| --- | --- | --- | --- |
| 0–2 | `base_ang_vel` | 3 | 机身角速度 |
| 3–5 | `projected_gravity` | 3 | 机身坐标系中的重力方向，反映倾斜 |
| 6–19 | `joint_pos` | 14 | 舵机关节相对默认姿态的位置，按配置加入编码器偏差 |
| 20–33 | `joint_vel` | 14 | 舵机关节速度 |
| 34–47 | `actions` | 14 | 上一步动作，帮助策略判断自身控制历史 |
| 48–50 | `command` | 3 | 期望前后速度、侧向速度、偏航角速度 |
| 51–54 | `head_command` | 4 | 颈部及头部的关节姿态指令 |
| 55–60 | `body_command` | 6 | 身体位置与姿态指令 |

前 48 维为本体感觉，后 13 维为命令。**“让机器人前进”是在修改命令输入，而不是直接给某个腿关节加速度。** 策略负责把命令转换成协调的关节动作。

这 61 维的顺序是部署接口：不同策略切换时必须保持一致。不使用的命令项也保留；训练中的小范围采样用来避免对应输入一直失活，部署时按策略约定填写。

### 输出：14 个关节位置动作

平地走路的动作缩放系数为 1.0，并以 HOME 默认关节姿态为偏置。忽略编码器偏差等修正时：

<equation>q^{target}_t = q^{HOME} + a_t</equation>

`a_t` 是网络输出的 14 维向量；旋转关节目标以弧度表示。目标继续经过动作/执行器处理，由 **BAM 舵机模型**计算实际作用力矩，再进入物理仿真。位置目标不会使关节瞬间到位。

14 个主动关节的顺序为：左腿 5 个 → 颈/头 4 个 → 右腿 5 个。带轮或带齿隙模型还有 `passive_*` 被动关节；读写主动关节必须按名称或 `_servo_joint_ids` 等辅助函数解析。

**代码入口：** 环境配置中的 `observations`、`actions`；`robot/microduck_constants.py` 的 `HOME_FRAME` 和 BAM 配置。

## 3. 一个环境步里发生什么

策略控制周期为 **0.02 s（50 Hz）**；物理步长为 **0.005 s**，每个动作持续 4 个物理子步。这是控制频率，和电脑一秒能算多少仿真步是两回事。

`ManagerBasedRlEnv.step(action)` 可以概括为：

```python
# 示意顺序，不是可直接替换的实现
处理动作：缩放、默认姿态偏置等
重复 4 次：执行器施加动作 → 推进物理 → 更新场景
检查终止条件 → 计算奖励与指标
重置结束的环境
刷新运动学 → 更新命令与事件 → 更新传感器
生成下一步观测
返回 observations, rewards, terminated, truncated, extras
```

不同职责由 Manager 分开管理：Observation / Action / Command / Reward / Termination / Event / Curriculum。环境配置决定有哪些项，Manager 调用这些项对应的函数。

**重置通常是自动完成的。** 一个机器人摔倒后重新开始，其他并行机器人继续采样。真正失败的 `terminated` 与时间上限的 `truncated` 要区分；后者通过包装器的 `time_outs` 进入 PPO 的价值补偿。

## 4. Actor 和 Critic 如何协作

本任务采用两个独立的 MLP，隐藏层均为 `512 → 256 → 128`，激活函数为 ELU。

| 网络 | 输入 | 输出 | 训练和部署中的作用 |
| --- | --- | --- | --- |
| Actor | 61 维观测 | 14 维动作均值；另有可学习的高斯分布标准差 | 训练时采样动作探索；播放和导出时默认用确定性输出 |
| Critic | 本任务默认 76 维观测 | 1 个价值估计 | 帮助计算优势与回报目标；部署不需要 |

Critic 额外看到机身线速度、脚接触、接触力等仿真信息，也保留较真实的传感状态。这样训练价值函数更容易，而 Actor 仍遵守真机可观测接口。这称为**非对称 Actor–Critic**。

两者都有观测归一化：训练中统计均值和尺度，再把观测送进网络。**训练后的 Actor 是“归一化器 + 网络”这一整体**，不是只剩几个 Linear 层。

**代码入口：** `microduck_velocity_env_cfg.py` 末尾的 `MicroduckRlCfg`；依赖库 `rsl_rl/models/mlp_model.py`。

## 5. 一次 PPO 迭代如何落到代码

`OnPolicyRunner.learn()` 反复执行三步。

**① 收集轨迹。** 当前策略同时驱动 N 个环境，每个环境走 24 个控制步。缓存观测、动作、奖励、结束标记、旧策略对动作的对数概率和 Critic 价值。

一次迭代的样本量为 **N × 24**。例如 64 环境得到 1,536 条 transition；4096 环境得到 98,304 条。每个机器人单轮采样对应 0.48 s 仿真时间。24 步是采样片段长度，不是完整 episode 长度。

**② 计算优势。** `PPO.compute_returns()` 从轨迹末尾反向递推 GAE，估计某个动作比价值函数预期好多少，并得到 Critic 的回报目标。当前参数 `gamma=0.99`、`lambda=0.95`。

**③ 更新网络。** `PPO.update()` 把这一批样本拆成 4 个 mini-batch，重复训练 5 个 epoch，通常每轮有 20 次优化器更新，然后清空采样缓存，使用新策略继续采样。

PPO 核心是限制新旧策略变化过大。记概率比为 ρ，优势为 A：

<equation>L_{policy}=-\mathbb{E}\left[\min\left(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\right)\right]</equation>

总损失由策略项、价值误差与熵项组成：

<equation>L = L_{policy} + c_v L_{value} - c_e H</equation>

当前 `epsilon=0.2`、价值损失系数 1.0、熵系数 0.01；Adam 初始学习率 0.001，根据 KL 自适应调整。熵项维持探索，Critic 拟合回报，策略项提升高优势动作的概率。

**真正要断点查看的四个函数：** `OnPolicyRunner.learn()`、`PPO.act()`、`PPO.compute_returns()`、`PPO.update()`。缓存实现位于 `rsl_rl/storage/rollout_storage.py`。

## 6. 奖励决定策略最后学成什么

训练器不知道“好看的走路”是什么，只优化环境返回的标量。默认奖励管理器把各项函数值乘以权重、再按控制步长积分：

<equation>r_t = \Delta t\sum_i w_i f_i(s_t,a_t,c_t)</equation>

当前走路配方主要由以下几组构成：

| 组别 | 代表项 | 目的与取舍 |
| --- | --- | --- |
| 任务跟踪 | `track_linear_velocity`、`track_angular_velocity` | 跟随速度命令，两项权重均为 2.0 |
| 保持姿态 | `upright`、`pose` | 保持直立和合理腿部姿态；过强会压制迈步 |
| 脚部行为 | `air_time`、`foot_clearance`、`foot_slip` | 鼓励摆腿、避免拖脚和过度打滑 |
| 平滑与约束 | `action_rate_l2`、`dof_pos_limits`、`self_collisions` | 减少动作跳变、触限和自碰撞 |
| 头部控制 | `head_pose_tracking`、`head_pose_bias` | 跟随头部指令；后者针对持续偏差，而非每一瞬间的摆动 |
| 身体命令接口 | `body_pose_tracking` | 走路任务权重为 0，保留输入接口供其他任务使用 |

**惩罚的符号必须看函数返回值。** 基础库 cost 返回正值，配负权重；部分自定义 penalty 已返回负值，应配正权重。只看函数名字或权重正负都不可靠。最终日志里每项惩罚的加权值应 ≤ 0。

例如“稳稳站住”也能获得直立、姿态和部分跟踪奖励。因此总 reward 增长不等于学会走路，必须同时看速度跟踪、实际位移、脚接触和视频。

**代码入口：** 权重、参数在任务 cfg；所有 Microduck 自定义 MDP 函数集中在 `tasks/mdp.py`。步态奖励也会调用 `mjlab` 自带函数。

## 7. 课程如何改变学习难度

课程分两类：**改变奖励权重**，或**改变命令、事件的采样范围**。它们改变训练问题，不是另一个优化算法。

走路任务逐步提高平滑惩罚、扩大头部命令范围、提高精确零速度命令的比例，并调整质心随机化范围。原地转向也有单独采样桶，避免它在连续随机命令中出现太少。

课程计数按环境控制步累计：**从头训练时，iteration × 24 ≈ common_step_counter**。这个计数不乘环境数；增加并行数量会增加数据量，但不会自动推迟课程阶段。

如果关键指标恰好在课程边界下降，先检查策略是否已学会上一阶段，再考虑延后难度或惩罚。运行中要通过 Manager 获取并修改 term cfg；Manager 初始化时复制配置，直接修改 `env.cfg` 可能不生效。

# 第二部分：工程性辅助

## 1. 环境隔离与日常使用

| 名称 | 实际作用 |
| --- | --- |
| `uv` | 安装依赖、创建/维护 Python 环境的工具 |
| `.venv` | 此项目实际使用的 Python 和第三方包；激活的是它 |
| `pyproject.toml` | 项目依赖、命令入口和包来源的声明 |
| `uv.lock` | 精确版本与来源的锁定记录，用于重建相同环境 |
| `.venv-tools/bin/uv` | 本机 uv 的安装位置，与训练环境分开 |

本机复用了系统 Python 3.12.3 来创建 `.venv`，无需每次运行重新下载 Python。编辑器标注的 `UV` 表示环境的管理方式。

**fish 日常入口：**

```fish
cd /home/elaina/project/microduck_rl
source .venv/bin/activate.fish
which python
list-envs
```

`which python` 应指向项目 `.venv/bin/python`。之后直接运行 `train`、`play`、`python`；退出用 `deactivate`。新终端重新激活。

需要安装依赖或跑测试时，再让当前终端找到 uv：

```fish
set -gx PATH "$PWD/.venv-tools/bin" $PATH
uv sync --locked
uv run --locked --with pytest pytest tests/
```

`uv sync --locked` 按锁文件同步依赖；它与训练、环境激活是不同操作。虚拟环境隔离 Python 包，GPU 驱动仍使用主机的。新主机是否可训练，检查实际 GPU 访问和 CUDA 版 PyTorch。

## 2. 最短的训练—观看—导出路径

以下命令在已激活的项目根目录执行。

### 先做小规模冒烟测试

```fish
train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.logger tensorboard \
  --agent.run-name smoke \
  --enable-nan-guard True
```

这验证环境能构建、步进、计算奖励和更新网络，**不用于证明已经学会走路**。通过后再根据显存和训练预算扩大环境数与迭代数；仓库主配方常用 4096 环境，步态通常需要数千次迭代。预算不是成功保证。

日志目录为 `logs/rsl_rl/velocity/<时间_名称>/`，含 checkpoint、TensorBoard 事件和 `params/` 配置。当前默认每 250 次迭代保存一次，结束时也保存；迭代从 0 编号，5000 次训练通常结束于 `model_4999.pt`。

### 加载实际 checkpoint 观看

```fish
# 本机已有的一个完整训练 checkpoint；换模型时修改这一行
set checkpoint /home/elaina/project/microduck_rl/logs/rsl_rl/velocity/2026-10-13_15-32-07_first-run/model_4999.pt

play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file $checkpoint --num-envs 1 --viewer native
```

`--viewer native` 打开 MuJoCo 桌面窗口，`--viewer viser` 使用浏览器界面。播放采用确定性 Actor，并加载归一化器。走路任务的 play 配置仍保留随机化，且外部推扰更频繁；评估普通步态时应明确推扰、命令和初始状态条件。

浏览器里想手动控制，启用 `Twist` 命令控制：`lin_vel_x` 为前后速度（m/s），`lin_vel_y` 为侧向速度（m/s），`ang_vel_z` 为转向角速度（rad/s）。仅填写数字但未启用手动控制，不能据此断定策略忽略命令。

### 用标准路径导出 ONNX

```fish
python scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file $checkpoint --num-envs 1 \
  --onnx-file artifacts/walk.onnx
```

输出接口应为 **`[1,61] → [1,14]`**，观测归一化嵌入图中。用 `scripts/infer_policy.py --walking ...` 做 CPU MuJoCo 部署演练，再考虑真机。用于硬件的文件始终走标准 exporter。

## 3. 如何读日志，如何判断效果

| 观察对象 | 能说明什么 | 不能单独说明什么 |
| --- | --- | --- |
| 总 reward | 总目标的优化趋势 | 是否真的跟随速度命令 |
| episode length | 活得更久或较少提前终止 | 是否运动，或只是在站立 |
| `Episode_Reward/<term>` | 各奖励项的加权贡献，受课程影响 | 权重为 0 时的原始行为质量 |
| 速度误差、摔倒、NaN | 跟踪、稳定性和数值健康 | 视频是否自然、是否能迁移到硬件 |
| value / surrogate / entropy | 优化过程与探索状态 | 动作是否满足任务定义 |

效果检查至少覆盖 **精确零命令、前进、后退、侧移、原地转向**，并明确头部/身体指令、推扰和初始姿态。先看实际 checkpoint 的视频，再统计速度误差、首次摔倒时间和接触状态。

**区分三个结果：程序跑完；策略在指定条件下完成动作；策略可部署到真机。** 这三个结论需要不同证据。不要因为 reward 上涨就直接加长训练或修改奖励；先测出失败具体发生在哪里。

## 4. Sim2real：这些工程项实际改变了学习问题

| 工程项 | 在哪里 | 为什么需要 |
| --- | --- | --- |
| MJCF 机器人与碰撞 | `robot/microduck/`、`microduck_constants.py` | 决定质量、惯量、关节和哪些部位能接触地面；走路与倒地任务不能随意共用碰撞模型 |
| BAM XL330 模型 | `actuator/friction_dr_bam.py`，依赖 `bam` | 模拟电压控制、摩擦、电池压降和动作延迟，使关节响应接近真机 |
| Domain Randomization（DR） | 任务 cfg 的事件与 `mdp.py` | 随机化质心、摩擦、质量等物理条件，避免只适应一台理想机器人 |
| 传感噪声与延迟 | 任务 cfg 的观测项 | 让 Actor 适应编码器、IMU 的误差和滞后 |
| Backlash 齿隙变体 | `tasks/backlash.py`、`add_backlash.py` | 用被动关节建模齿隙，编码器读数也映射到输出侧 |
| NaN 防护 | `mdp.py` 开头补丁、终止项和 `_safe` 观测 | 避免单个异常状态污染奖励、优势和网络参数 |

需要守住的接口和物理约定：

- **摩擦 DR 调整 BAM 的 `friction_scale`。** BAM 下仅随机化 `dof_frictionloss` 会成为无效操作；独立环境需要注册 `expand_bam_friction_fields`。
- **DR 每次恢复基准再采样，不能跨 reset 累积。** 否则训练越久，物理世界越偏离最初定义。
- **观测与奖励使用一致的测量视角。** 例如齿隙编码器读的是输出侧，相关跟踪奖励也要按同一侧计算。
- **训练与部署统一 61 维接口、关节顺序、HOME、动作缩放和滤波约定。** 当前策略无动作低通滤波，不能只在一端额外加 EMA。

`mdp.py` 在导入时还会替换依赖库的部分方法，因此启动日志出现 `Patches ... active`。调试 PPO 或奖励数值时，既要读上游函数，也要看这些实际生效的补丁。NaN 防护是故障隔离，不能把它当作物理正确的证明。

## 5. 恢复训练、迁移和发布

**Resume 是继续同一个任务。** checkpoint 除了网络权重，还保存优化器、归一化统计、迭代数和 `common_step_counter`，后者用于继续课程。通过 `--agent.resume True`、`--agent.load-run` 与 `--agent.load-checkpoint` 指定原运行和模型。

**Warm start 是把已有策略作为新任务起点。** 原计数若直接恢复，会把新任务的课程跳到后期。仓库提供 `MICRODUCK_WARM_START=1` 来重置两个计数，同时保留权重、归一化器和优化器；还需检查源任务课程与归一化器是否兼容，不能把所有 checkpoint 视为可互换。

`train_hook.py` / `hf_jobs.py` 支持通过 `--hf-jobs` 把训练提交到 Hugging Face Jobs，属于训练运行位置的切换，不改变 PPO 基本流程。远程重建依赖也依赖 `pyproject.toml` 和锁文件。

`publish/` 负责验证 ONNX 的形状、有限输出等，并上传 **policy.onnx + schema-2 manifest.json + README**。manifest 记录策略用途、指令约定和训练来源；真机运行由另一个仓库 `pollen-robotics/microduck` 负责。文件能导出或上传，不代表动作已经通过部署验证。

## 6. 改代码时从哪里入手

| 目标 | 优先修改/阅读位置 |
| --- | --- |
| 调整现有走路奖励、命令和课程 | `tasks/microduck_velocity_env_cfg.py` |
| 增加一种奖励、重置或观测计算 | `tasks/mdp.py`；配套 `tests/` |
| 新建任务并出现在 `list-envs` | 选相近任务 cfg 作为模板，再注册到 `tasks/__init__.py` |
| 修改质量、关节、碰撞和执行器 | `robot/microduck_constants.py`、MJCF、`actuator/` |
| 修改通用 PPO 数学更新 | 依赖 `rsl_rl/algorithms/ppo.py`；注意本仓库导入补丁 |
| 修改模型导出或分发 | `export.py`、`scripts/export.py`、`publish/` |

建议阅读顺序：**走路 cfg → `MicroduckRlCfg` → `env.step()` → `Runner.learn()` → `PPO.update()` → 按函数名查 `mdp.py` → 导出流程。** 第一次不用从几千行 `mdp.py` 的第一行一路读到末尾。

增加新任务时，先在仿真验证目标姿态和接触条件，再写配置/函数测试，最后做 64 环境、5 次迭代的冒烟测试。明确任务指标后再开长训练。

**代码与约定来源：** [当前项目](https://github.com/Elaina-rascal/microduck_rl)、[官方 Microduck RL](https://github.com/pollen-robotics/microduck_rl)、仓库 `AGENTS.md`，以及上述本机已安装依赖源码。本说明按本次核对版本编写，版本升级后重新核对调用链和接口。
