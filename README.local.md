# 本机 Go2 地形项目

目录：`/home/clt/project_rl/unitree_rl_gym`。日期：2026-11-12，Asia/Shanghai。

已采用 **48 维普通 MLP＋独立地形课程＋48 维 MuJoCo 控制器**。原课程配置、控制器、策略/地形导出工具和统一 MuJoCo 入口已接入；本机环境、纯推理回放与键盘依赖已验证。当前没有训练或自动训练任务。

## 环境与入口

| 项目 | 本机配置 |
| --- | --- |
| 上游 Unitree 提交 | `276801e46c5d433564f24658bac64f254b7d2d4b` |
| GPU / 驱动 | RTX 4060 Laptop 8 GB / 550.163.01 |
| Python | 本项目 `.venv`，3.8.20 |
| PyTorch / wheel CUDA | 2.0.1+cu118 / 11.8 |
| Isaac Gym / rsl_rl | 本项目 Preview 4 SDK / v1.0.2 |
| NumPy / MuJoCo | 1.23.5 / 3.2.3 |
| 键盘 | pynput 1.7.7、evdev 1.9.2、python-xlib 0.33 |

解释器和依赖在本项目，Conda 管理器复用 `/home/clt/legged_gym/.tools/miniforge3`。未修改驱动、系统 CUDA、全局 shell 配置或 UniLab。未发现系统 `nvcc`；驱动支持上限不等于 wheel 使用的 CUDA runtime。上游 NumPy 固定版本已本地调整为 `>=1.23.5,<1.24`，`pip check` 通过。

```bash
cd /home/clt/project_rl/unitree_rl_gym
./bin/run python -c 'import isaacgym, torch, legged_gym; print(legged_gym.__file__); print(torch.cuda.is_available())'

# 可选的交互激活
source /home/clt/project_rl/unitree_rl_gym/activate.sh
conda deactivate
```

`bin/run` 设置本项目解释器、导入路径、动态库、C++ 编译器、SDK 缓存和 include 路径。冻结依赖见 `verification/requirements-freeze.txt`。键盘依赖已安装；重建时若 evdev 遇到 Conda 编译器与系统 Linux 头文件不匹配，可仅在该次安装中指定：

```bash
./bin/run env C_INCLUDE_PATH=/usr/include:/usr/include/x86_64-linux-gnu \
  python -m pip install -e '.[sim2sim]' --no-build-isolation
```

## 原源码接入范围

- 原 `go2_terrain_config(1).py` 已逐字接入 `legged_gym/envs/go2/go2_terrain_config.py`。
- 原 `go2_terrain.py` 已接入，补入本机评估工具需要的冻结等级、固定指令、终止前快照、奖励分量记录和 reset 后首帧状态刷新。适配差异见 `verification/source_alignment/go2_terrain_adapter.diff`。
- 原 `deploy_go2.py`、`export_go2_policy.py`、`export_go2_terrain.py`、`deploy_mujoco.py` 均逐字接入。
- 原 `go2.yaml` 保留 48 维布局、kp=20/kd=0.5、动作比例 0.25、关节顺序、23.7/35.55 N·m 限幅及按住运动/松开停止；默认策略路径指向已有短程 `model_3` 验收策略。
- 基础环境保留官方 48 维版本；未混入 HIM、270 维历史观测、特权观测、7 项 step 返回值或速度课程。
- 原 `terrain(1).py`、`play(1).py` 不符合日志最后状态，继续保留按日志补建的纯平地生成、固定地图评估与预览工具。生成器已支持原配置的 `flat_rows`、`roughness_range`。

提供的文件、哈希在 `verification/source_files/`；接入前备份在 `verification/before_source_alignment/`；逐字一致性在 `verification/source_alignment/source_identity.json`。原始最终地形生成器、专用回放工具与 CLT/Lain XML/网格仍未提供，相应部分不能声称逐字恢复。

## 地形预览：不训练

```bash
# 全等级地图，站立 PD
./bin/run python legged_gym/scripts/preview_terrain.py \
  --mode atlas --num-envs 200 --duration 20

# 4096 台从纯平地启动
./bin/run python legged_gym/scripts/preview_terrain.py \
  --mode start --num-envs 4096 --headless --duration 2

# heightfield 碰撞表示
./bin/run python legged_gym/scripts/preview_terrain.py \
  --mode atlas --mesh-type heightfield --num-envs 200 --duration 20
```

地图为 10×20，每块 8×8 m，level 0 全列纯平地。列 0～1 缓坡、2～3 碎石坡、4～10 低中心台阶（向外上）、11～15 高中心台阶（向外下）、16～19 障碍块。

原配置要求连续 3 个合格平地回合才能晋级：实际至少 3 秒、累计指令距离至少 2 米、正向进展至少 50%、xy/yaw 跟踪分数均至少 0.75、无物理失败。地形阶段阈值 0.60，距中心 3.5 米处截断，按表现升降级；物理失败优先于 timeout。

## Isaac 策略回放：不更新权重

```bash
./bin/run python legged_gym/scripts/play.py \
  --task go2_terrain --load_run Nov12_19-35-04_reconstruction_acceptance --checkpoint 3 \
  --num_envs 1 --terrain-level 1 --terrain-column 5 \
  --cmd 0.4 0 0 --duration 20
```

`--terrain-level 0` 看平地，`-1` 覆盖所有等级。地图生成后冻结等级，固定指令从第一帧观测生效；不创建优化器。报告使用终止/reset 前状态，区分身体失败、边界完成与超时。事件数包含多次 reset，不能当作独立通过率。

**当前 `model_3` 只做过短程验收更新，不能当作能行走的策略。原 `model_1500.pt` 未提供。**

## MuJoCo 平地与地形回放

```bash
# 平地，按住 WASD/方向键运动、松开归零
./bin/run python deploy/deploy_mujoco/deploy_mujoco.py go2.yaml --duration 20

# 等价独立入口
./bin/run python deploy/deploy_mujoco/deploy_go2.py --duration 20

# 已导出的低级上台阶
./bin/run python deploy/deploy_mujoco/deploy_go2.py \
  --config /home/clt/project_rl/unitree_rl_gym/verification/source_alignment/mujoco/level1_col5/go2.yaml \
  --duration 20

# 无窗口固定指令
./bin/run python deploy/deploy_mujoco/deploy_go2.py \
  --config /home/clt/project_rl/unitree_rl_gym/verification/source_alignment/mujoco/level1_col5/go2.yaml \
  --headless --cmd 0.4 0 0 --duration 6 --fast --report /tmp/go2_mujoco.json
```

W/S 前后，A/D 横移，E/Q 左右转，`[`/`]` 调倍率，空格停止并要求重新按方向键，P 暂停，R 重置，Esc 退出。`--headless`、`--fast` 必须搭配 `--cmd`。

六个场景在 `verification/source_alignment/mujoco/`：level0_col5、level1_col5、level4_col0/5/12/16。报告 height 为世界 z，tilt 单位为度；负中心台阶中的负世界 z 不等于掉入地面。块外没有 Isaac 的边界 reset。

导出自有检查点与指定地形：

```bash
./bin/run python deploy/deploy_mujoco/export_go2_policy.py \
  --checkpoint /实际路径/model_N.pt --output /实际输出目录/policy_1.pt
./bin/run python deploy/deploy_mujoco/export_go2_terrain.py \
  --level 4 --column 5 --seed 1 \
  --policy /实际输出目录/policy_1.pt --output /实际输出目录/level4_col5
```

公开 MuJoCo Menagerie Go2 位于 `resources/robots/go2/mujoco/`（质量 15.206408 kg），仅给现有 free joint 补名 `root` 以匹配原控制器，保留原惯量/接触/阻尼。旧 `mjcf/` 是兼容早期导出路径的符号链接。原 CLT/Lain XML 缺失，质量相同不能证明动力学逐项相同；高度场也不包含 Isaac 的垂直边缘修正。

## 验收与训练状态

本轮结果见 `verification/source_alignment/REPORT.md`，总记录见 `verification/REPORT.md`。本轮只做配置/接口、站立 PD、推理、导出、键盘钩子和 GUI 验证，没有 PPO 更新，原检查点哈希不变。

此前误启动的新训练保留在 `logs/go2_terrain/Nov12_19-41-41_reconstructed_flat_to_terrain/`：记录 96 次更新，最后保存 `model_50.pt`。它属于此前复建版本，不是原实验模型，也不作为本次源码对齐的训练成果。用户叫停后未恢复，自动启动/等待脚本已删除。

`./bin/status-go2-terrain` 只读显示状态。未来只有用户明确决定训练时才使用前台 `bin/train-go2-terrain`。恢复检查点不会恢复完整仿真状态、课程分布或随机数状态。

短程验收不证明可靠行走、地形通过、收敛或跨引擎性能等价。8 GB 显卡应顺序运行大规模 Isaac 实例。未操作实物机器人。
