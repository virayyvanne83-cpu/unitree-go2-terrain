# Go2 平地 MuJoCo sim2sim 与键盘控制

和 `g1.yaml` / `h1.yaml` 一样的方式：一个 MJCF、一个 YAML、一个策略文件。
Go2 单独一个脚本只是因为它的 48 维观测（机体系线速度、无步态相位）和人形的 47 维布局不同。

## 1. 环境

```bash
cd /home/clt/project_rl/unitree_rl_gym
source /home/clt/project_rl/unitree_rl_gym/activate.sh
```

键盘需要 `pynput`，未安装时执行 `python -m pip install -e '.[sim2sim]'`。

## 2. 导出自己的模型

```bash
python deploy/deploy_mujoco/export_go2_policy.py \
  --checkpoint logs/go2_terrain/Nov12_19-35-04_reconstruction_acceptance/model_3.pt \
  --output logs/go2_terrain/Nov12_19-35-04_reconstruction_acceptance/exported/model_3/policy_1.pt
```

不指定 `--output` 时写到 `go2.yaml` 的 `policy_path`。改过激活函数要加 `--activation relu`（或 `tanh/selu/lrelu/sigmoid`）。

## 3. 键盘运行

```bash
python deploy/deploy_mujoco/deploy_mujoco.py go2.yaml
```

等价独立入口：`python deploy/deploy_mujoco/deploy_go2.py`。策略不是默认路径时加 `--policy <路径>`。

| 按键 | 动作 |
| --- | --- |
| W / ↑，S / ↓ | 前进、后退 |
| A / ←，D / → | 左移、右移 |
| E，Q | 左转、右转 |
| ]，[ | 速度倍率 ±0.1，范围 0.1～1.5 |
| 松开方向键 | 该方向指令归零 |
| 空格 | 指令清零，需重新按方向键 |
| P | 暂停、继续 |
| R | 重置位姿 |
| Esc | 退出 |

基础速度 vx=0.4 m/s、vy=0.4 m/s、yaw=1 rad/s，乘倍率后裁剪到 ±1。

## 4. 固定指令 / 无界面

```bash
python deploy/deploy_mujoco/deploy_mujoco.py go2.yaml \
  --policy logs/go2_terrain/Nov12_19-35-04_reconstruction_acceptance/exported/model_3/policy_1.pt \
  --headless --cmd 0.4 0 0 --duration 8 --fast \
  --report /tmp/go2_eval/fwd.json
```

`--headless`、`--fast` 必须搭配 `--cmd`。报告里的 `mean_vx_vy_yaw_height_tilt` 依次是机体系 vx、vy、yaw 角速度、base 高度、倾角（度）。

## 5. 接口

| 观测切片 | 内容 |
| --- | --- |
| 0:3 | 机体系线速度 ×2 |
| 3:6 | 机体系角速度 ×0.25 |
| 6:9 | 机体系重力方向 |
| 9:12 | vx/vy/yaw 指令 ×[2,2,0.25] |
| 12:24 | 关节角减默认角 |
| 24:36 | 关节速度 ×0.05 |
| 36:48 | 上一次裁剪后的动作 |

- 动作 = `default_angles + action_scale(0.25) × action`，`kp=20`、`kd=0.5`，力矩按 `23.7 / 35.55 N·m` 限幅。
- 物理步 0.005 s，每步算 PD，4 步跑一次策略 = 50 Hz。
- MuJoCo 四元数是 wxyz；线速度从世界系转到机体系，free-joint 角速度本身就是机体系。
- 初始 root 高度 0.42 m（会先落到站立高度），默认关节角与训练配置一致。
- 默认关闭机器人自碰撞，保留与地面接触。

## 6. 与 Isaac Gym 的差别

观测、kp/kd/action_scale、默认角、力矩限制和时间轴按 Go2 配置设置。
本机使用公开 MuJoCo Menagerie Go2，总质量约 15.206408 kg；仅将 free joint 命名为 root 以适配原控制器，没有启用逐刚体惯性同步。原 CLT/Lain 模型未提供，不能声称惯量或接触逐项相同。
接触、摩擦、积分和约束求解算法也不同；不能保证两个引擎轨迹逐帧一致。

## 7. 模型与资源

`resources/robots/go2/mujoco/` 下的 `go2.xml`、`scene.xml` 和视觉网格来源见该目录的 `PROVENANCE.md` 与 `LICENSE`。

## 8. 地形策略与地形场景

`go2_terrain` 仍为 48 维接口，导出模型后可使用本控制器；仅替换策略不会自动把平地变成台阶。
正式工具 `export_go2_terrain.py` 可按 level/column/seed 导出训练地图中的单块高度场及正确出生高度。
用 `deploy_go2.py --config /绝对路径/地形目录/go2.yaml` 启动该场景；原键盘控制继续可用。

当前默认策略是旧版短程 model_3 验收权重，不能证明已学会行走。原 model_1500 未提供。
完整本机命令与六组场景见 [本机说明](../../README.local.md)。
MuJoCo 高度场不含 Isaac 垂直边缘修正，也不执行课程/边界自动 reset；请在块内短程检查，或 R 人工重置。
