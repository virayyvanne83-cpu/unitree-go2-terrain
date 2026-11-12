# Go2 地形课程与 48 维 MuJoCo 回放

本项目基于 Unitree 官方 `unitree_rl_gym`，新增独立 `go2_terrain` 任务。采用 48 维普通 ActorCritic MLP、纯平地起步的地形课程，以及 48 维 MuJoCo 控制器。原 `go2` 任务保留，不使用 HIM、历史观测或速度课程。

## 功能

- 4096 并行环境，10×20 有序地形地图；第 0 行全部为纯平地。
- 每个环境按实际行走时间、指令距离、前进程度、速度跟踪与失败状态独立升降级。
- 48 维 actor/critic 输入，独立 `[512,256,128]` ELU MLP，PD 为 kp=20/kd=0.5。
- trimesh 与 heightfield、地形预览、指定等级/列/指令的纯推理回放。
- TorchScript 策略导出、单块地形导出和原生 MuJoCo 射线高度核对。
- MuJoCo 按关节名称映射，质心速度修正，23.7/35.55 N·m 力矩限幅，按住移动/松开停止的键盘控制。

## 安装

已验证的组合：Ubuntu 22.04、Python 3.8.20、Isaac Gym Preview 4、rsl_rl v1.0.2、PyTorch 2.0.1+cu118、NumPy 1.23.5、MuJoCo 3.2.3。该组合在 RTX 4060 Laptop、驱动 550.163.01 上通过运行检查；不要求修改系统 CUDA Toolkit。

先进入克隆后的本项目根目录。需要自行安装 Conda/Miniforge，并从 NVIDIA 官方获取 Isaac Gym Preview 4，将 SDK 放在本项目 `isaacgym/`。SDK 不随此仓库发布。

```bash
conda create -p "$PWD/.venv" -c conda-forge \
  python=3.8.20 pip=24.3.1 setuptools=69.5.1 wheel=0.45.1 gxx_linux-64=11 -y
conda activate "$PWD/.venv"

git clone --branch v1.0.2 --depth 1 https://github.com/leggedrobotics/rsl_rl.git rsl_rl
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-local.txt
python -m pip install --no-deps -e isaacgym/python -e rsl_rl -e .
python -m pip install -e '.[sim2sim]'
python -m pip check
```

本项目 `bin/run` 自动定位仓库与 `.venv`，设置 SDK 导入路径、动态库和本地编译器。`activate.sh`、`README.local.md` 和验收说明中包含原测试机器路径，迁移时请使用当前克隆目录替换；推荐直接使用 `bin/run`。

如果 evdev 构建时遇到 Conda 与系统 Linux 头文件不一致，可在该次安装命令中使用：

```bash
./bin/run env C_INCLUDE_PATH=/usr/include:/usr/include/x86_64-linux-gnu \
  python -m pip install -e '.[sim2sim]' --no-build-isolation
```

## 先预览，再选择是否训练

下列命令只做站立 PD 预览，不更新网络：

```bash
./bin/run python legged_gym/scripts/preview_terrain.py --mode atlas --num-envs 200 --duration 20
./bin/run python legged_gym/scripts/preview_terrain.py --mode start --num-envs 4096 --headless --duration 2
```

训练必须明确手动启动；项目没有自动训练任务。检查点与 TensorBoard 事件保存在 `logs/`，不随源码仓库发布。

```bash
./bin/run python legged_gym/scripts/train.py --task go2_terrain \
  --num_envs 4096 --headless --run_name flat_to_terrain
```

## 使用自己的检查点回放

```bash
./bin/run python legged_gym/scripts/play.py --task go2_terrain \
  --load_run YOUR_RUN --checkpoint 1500 --num_envs 20 \
  --terrain-level 0 --cmd 0.4 0 0 --duration 20

./bin/run python deploy/deploy_mujoco/export_go2_policy.py \
  --checkpoint /absolute/path/model_N.pt --output /absolute/path/policy_1.pt

./bin/run python deploy/deploy_mujoco/deploy_mujoco.py go2.yaml \
  --policy /absolute/path/policy_1.pt --duration 20

./bin/run python deploy/deploy_mujoco/export_go2_terrain.py \
  --level 1 --column 5 --seed 1 --policy /absolute/path/policy_1.pt \
  --output /absolute/path/level1_col5

./bin/run python deploy/deploy_mujoco/deploy_go2.py \
  --config /absolute/path/level1_col5/go2.yaml --headless \
  --cmd 0.4 0 0 --duration 6 --fast --report /tmp/go2_report.json
```

`configs/go2.yaml` 中默认策略地址是原测试机器的验收运行名，仓库不包含该策略；请显式使用 `--policy` 或修改配置。WASD/方向键移动，E/Q 转向，松开停止，空格清零，P 暂停，R 重置，Esc 退出。

## 验证范围和来源

本机已验证 4096×48 观测、两种地形碰撞表示、课程规则、固定地形推理、控制器接口、地形导出射线高度、有限状态和力矩限制。短程检查不证明可靠行走、复杂地形通过、收敛或跨引擎等价；原 model_1500 权重没有提供。

`verification/` 保留可重复运行的检查工具。原始日志、历史备份、机器环境文件、生成场景与权重不纳入源码发布；本机说明中的历史结果路径可能只在原测试机器存在。

Unitree 上游基线为 `276801e46c5d433564f24658bac64f254b7d2d4b`。MuJoCo Go2 资源来自 MuJoCo Menagerie，来源和修改见 `resources/robots/go2/mujoco/PROVENANCE.md`，许可证保留。原 CLT/Lain XML 缺失，不能声称与原模型逐项相同。
