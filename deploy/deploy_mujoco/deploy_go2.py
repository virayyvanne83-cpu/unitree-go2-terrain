"""Go2 flat-ground MuJoCo sim2sim: hold WASD/arrows to move, E/Q to turn, release to stop.

Keyboard: [/] speed, Space stop, P pause, R reset, Esc exit.
Headless example: --headless --cmd 0.4 0 0 --duration 5 --fast

This is the same lightweight pattern as the g1/h1 configs: one MJCF, one YAML,
one policy file. Go2 needs its own script only because its 48-dim observation
(cartesian base velocity, no gait phase) differs from the humanoid 47-dim layout.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "configs/go2.yaml"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    if not np.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}")


def resolve(path):
    return Path(str(path).replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT))).expanduser().resolve()


def rotation(quat_wxyz):
    matrix = np.empty((3, 3))
    mujoco.mju_quat2Mat(matrix.reshape(-1), quat_wxyz)
    return matrix


def rpy(matrix):
    return (math.atan2(matrix[2, 1], matrix[2, 2]),
            math.atan2(-matrix[2, 0], math.hypot(matrix[2, 1], matrix[2, 2])),
            math.atan2(matrix[1, 0], matrix[0, 0]))


class KeyboardState:
    """Thread-safe press/release state; discrete commands ignore key autorepeat."""
    def __init__(self):
        self._lock = threading.Lock()
        self.held = set()
        self.blocked = set()
        self.events = []
        self.listener = None

    @staticmethod
    def code(key):
        char = getattr(key, "char", None)
        return char.lower() if char else getattr(key, "name", None)

    def press(self, code):
        if code is None:
            return
        with self._lock:
            if code in self.held:
                return
            self.held.add(code)
            if code in ("esc", "space", "p", "r", "[", "]"):
                self.events.append(code)

    def release(self, code):
        with self._lock:
            self.held.discard(code)
            self.blocked.discard(code)

    def snapshot(self):
        with self._lock:
            keys, events = self.held - self.blocked, self.events[:]
            self.events.clear()
            if "space" in keys or "space" in events:
                # Forget held directions too: Space actually stops until a new press.
                self.blocked.update(self.held & {"w", "s", "a", "d", "e", "q", "up", "down", "left", "right"})
                keys = set()
            return keys, events

    def start(self):
        try:
            from pynput import keyboard
            self.listener = keyboard.Listener(
                on_press=lambda key: self.press(self.code(key)),
                on_release=lambda key: self.release(self.code(key)))
            self.listener.start()
            self.listener.wait()
        except Exception as exc:
            self.stop()
            raise RuntimeError("Keyboard unavailable; install .[sim2sim] and use a desktop, or pass --cmd VX VY YAW") from exc

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener.join(timeout=1)
            self.listener = None


class MotionCommands:
    def __init__(self, cfg):
        self.base = np.array([cfg["cmd_base"][key] for key in ("vx", "vy", "yaw")])
        self.limits = np.array([cfg["cmd_range"][key] for key in ("vx", "vy", "yaw")])
        finite(self.limits, "command limits")
        require(self.limits.shape == (3, 2) and np.all(self.limits[:, 0] <= self.limits[:, 1]), "Invalid command ranges")
        self.scale = 1.0
        self.scale_step = cfg["cmd_scale_step"]
        self.scale_bounds = cfg["cmd_scale_range"]
        self.paused = False

    def clamp(self, command):
        command = np.asarray(command, dtype=float)
        require(command.shape == (3,), "Expected three motion commands")
        finite(command, "command")
        return np.clip(command, self.limits[:, 0], self.limits[:, 1])

    def update(self, keys, events):
        for event in events:
            if event in ("[", "]"):
                self.scale = float(np.clip(self.scale + (1 if event == "]" else -1) * self.scale_step,
                                           *self.scale_bounds))
            elif event == "p":
                self.paused = not self.paused
        direction = np.array([int(bool(keys & {"w", "up"})) - int(bool(keys & {"s", "down"})),
                              int(bool(keys & {"a", "left"})) - int(bool(keys & {"d", "right"})),
                              int("e" in keys) - int("q" in keys)])
        return self.clamp(direction * self.base * self.scale)


class Go2Simulation:
    def __init__(self, cfg, policy_path):
        self.cfg = cfg
        require(cfg["num_obs"] == 48 and cfg["num_actions"] == 12,
                "Only 48-observation/12-action Go2 policies are supported")
        self.dt = float(cfg["simulation_dt"])
        self.decimation = int(cfg["control_decimation"])
        require(math.isfinite(self.dt) and self.dt > 0 and self.decimation > 0, "Invalid simulation timing")
        self.model = mujoco.MjModel.from_xml_path(str(resolve(cfg["xml_path"])))
        self.model.opt.timestep = self.dt
        require((self.model.nq, self.model.nv, self.model.nu) == (19, 18, 12),
                "Expected a free-root 12-joint Go2 model")
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        require(root >= 0 and self.model.jnt_type[root] == mujoco.mjtJoint.mjJNT_FREE and
                self.model.jnt_qposadr[root] == 0 and self.model.jnt_dofadr[root] == 0,
                "Expected free joint 'root' at qpos/qvel offset 0")
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in cfg["joint_names"]]
        require(len(joint_ids) == 12 and len(set(joint_ids)) == 12 and min(joint_ids) >= 0,
                "Missing or duplicate Go2 joint names")
        require(all(self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in joint_ids),
                "Go2 leg joints must be hinges")
        self.qidx = self.model.jnt_qposadr[joint_ids].copy()
        self.vidx = self.model.jnt_dofadr[joint_ids].copy()
        actuators = []
        for joint in joint_ids:
            matches = np.flatnonzero((self.model.actuator_trnid[:, 0] == joint) &
                                      (self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
            require(len(matches) == 1, "Each Go2 joint needs exactly one joint motor")
            actuators.append(int(matches[0]))
        self.aidx = np.array(actuators)
        require(np.allclose(self.model.actuator_gear[self.aidx, 0], 1), "Expected unit motor gear ratios")
        for field in ("default_angles", "kps", "kds", "torque_limits"):
            value = np.asarray(cfg[field], dtype=float)
            require(value.shape == (12,), f"{field} must have 12 entries")
            finite(value, field)
            setattr(self, field, value)
        require(np.all(self.kps > 0) and np.all(self.kds >= 0) and np.all(self.torque_limits > 0),
                "Invalid PD gains or torque limits")
        self.action_scale = float(cfg["action_scale"])
        self.clip_actions = float(cfg["clip_actions"])
        self.clip_observations = float(cfg["clip_observations"])
        self.cmd_scale = np.asarray(cfg["cmd_scale"], dtype=float)
        require(self.cmd_scale.shape == (3,), "cmd_scale must have 3 entries")
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setConst(self.model, self.data)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        if not cfg["self_collisions"]:
            # Keep robot<->ground, drop robot<->robot contacts (matches training).
            robot = (self.model.geom_bodyid != 0) & ((self.model.geom_contype | self.model.geom_conaffinity) != 0)
            ground = self.model.geom_bodyid == 0
            self.model.geom_contype[robot], self.model.geom_conaffinity[robot] = 1, 0
            self.model.geom_contype[ground], self.model.geom_conaffinity[ground] = 0, 1
        for field, scale_key in (("lin_vel_scale", "lin_vel_scale"), ("ang_vel_scale", "ang_vel_scale"),
                                 ("dof_pos_scale", "dof_pos_scale"), ("dof_vel_scale", "dof_vel_scale")):
            require(float(cfg[scale_key]) >= 0, f"Invalid {scale_key}")
        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        with torch.inference_mode():
            probe = self.policy(torch.zeros(1, 48))
        require(isinstance(probe, torch.Tensor) and tuple(probe.shape) == (1, 12),
                "Policy must return a [1,12] action tensor for [1,48] observations")
        finite(probe.numpy(), "policy probe")
        self.policy_calls = 0
        self.command = np.zeros(3)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0, 0, self.cfg["init_height"]]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self.qidx] = self.default_angles
        mujoco.mj_forward(self.model, self.data)
        self.action = np.zeros(12)
        self.target = self.default_angles.copy()
        self.control_steps = 0
        self.policy_steps = 0
        self.last_torque = np.zeros(12)

    def com_velocity(self):
        matrix = rotation(self.data.qpos[3:7])
        return matrix.T @ self.data.qvel[:3] + np.cross(self.data.qvel[3:6], self.model.body_ipos[self.base_id])

    def observe(self, command):
        matrix = rotation(self.data.qpos[3:7])
        obs = np.concatenate((self.com_velocity() * self.cfg["lin_vel_scale"],
                              self.data.qvel[3:6] * self.cfg["ang_vel_scale"],
                              -matrix[2],
                              command * self.cmd_scale,
                              (self.data.qpos[self.qidx] - self.default_angles) * self.cfg["dof_pos_scale"],
                              self.data.qvel[self.vidx] * self.cfg["dof_vel_scale"],
                              self.action)).astype(np.float32)
        finite(obs, "48-dimensional observation")
        return np.clip(obs, -self.clip_observations, self.clip_observations)

    def step(self, command):
        if self.control_steps % self.decimation == 0:
            self.policy_calls += 1
            observation = self.observe(command)
            with torch.inference_mode():
                raw = self.policy(torch.from_numpy(observation).unsqueeze(0)).numpy()
            require(raw.shape == (1, 12), "Policy returned an invalid action shape")
            finite(raw, "policy action")
            self.action = np.clip(raw[0], -self.clip_actions, self.clip_actions)
            self.target = self.default_angles + self.action * self.action_scale
        torque = self.kps * (self.target - self.data.qpos[self.qidx]) - self.kds * self.data.qvel[self.vidx]
        finite(torque, "PD torque")
        self.last_torque = np.clip(torque, -self.torque_limits, self.torque_limits)
        self.data.ctrl[self.aidx] = self.last_torque
        mujoco.mj_step(self.model, self.data)
        finite(self.data.qpos, "qpos")
        finite(self.data.qvel, "qvel")
        if any(warning.number for warning in self.data.warning):
            raise FloatingPointError("MuJoCo reported a simulation warning; inspect policy and dynamics")
        self.control_steps += 1
        if self.control_steps % self.decimation == 0:
            self.policy_steps += 1

    def physical_settings(self):
        return dict(mass_kg=float(self.model.body_mass.sum()), physics_dt=self.dt,
                    policy_period=self.dt * self.decimation, self_collisions=self.cfg["self_collisions"],
                    terrain=self.cfg.get('terrain'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", type=Path, help="Override go2.yaml's policy_path")
    parser.add_argument("--cmd", nargs=3, type=float, metavar=("VX", "VY", "YAW"), help="Fixed command; disables keyboard")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, help="Simulation seconds, excluding pauses")
    parser.add_argument("--fast", action="store_true", help="Run fixed-command simulation without wall-clock pacing")
    parser.add_argument("--report", type=Path, help="Write run summary JSON")
    args = parser.parse_args(argv)
    if (args.headless or args.fast) and args.cmd is None:
        parser.error('--headless/--fast requires --cmd VX VY YAW')
    return args


def run(args):
    cfg = yaml.safe_load(args.config.read_text())
    duration = args.duration if args.duration is not None else cfg["simulation_duration"]
    require(math.isfinite(duration) and duration > 0, "Duration must be finite and positive")
    policy_path = resolve(args.policy or cfg["policy_path"])
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"Policy not found: {policy_path}. First run export_go2_policy.py --checkpoint /path/to/model_N.pt")
    torch.set_num_threads(1)
    sim = Go2Simulation(cfg, policy_path)
    motion = MotionCommands(cfg)
    command = motion.clamp(args.cmd if args.cmd is not None else [0, 0, 0])
    keyboard = KeyboardState() if args.cmd is None else None
    settings = sim.physical_settings()
    print("[sim2sim] policy:", policy_path, flush=True)
    print("[sim2sim] physics:", json.dumps(settings), flush=True)
    target_steps = int(math.ceil(duration / sim.dt))
    count = 0
    changes, event_counts = [], {"speed": 0, "pause": 0, "reset": 0, "stop": 0, "quit": 0}
    stats = np.zeros(5)
    samples = 0
    min_height, max_torque_ratio = float(sim.data.qpos[2]), 0.0
    previous, reason = None, "duration"
    context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(sim.model, sim.data)
    try:
        with context as viewer:
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                        (5.0, 135, -30) if cfg.get('terrain') else (2.5, 135, -20))
            if keyboard is not None:
                keyboard.start()
                print("[sim2sim] 键盘已就绪: WASD/方向键移动, E左转/Q右转, [/]调速, 空格停, P暂停, R重置, Esc退出", flush=True)
            while count < target_steps:
                start = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    reason = "window_closed"
                    break
                if keyboard is not None:
                    keys, events = keyboard.snapshot()
                    command = motion.update(keys, events)
                    if events:
                        print(f"[sim2sim] controls={events} scale={motion.scale:.2f} paused={motion.paused}", flush=True)
                    for event, name in (("[", "speed"), ("]", "speed"), ("p", "pause"), ("r", "reset"),
                                        ("space", "stop"), ("esc", "quit")):
                        event_counts[name] += events.count(event)
                    if "r" in events:
                        sim.reset()
                    if "esc" in events:
                        reason = "keyboard_exit"
                        break
                changed = previous is None or not np.array_equal(previous, command)
                if changed:
                    previous = command.copy()
                    if len(changes) < 1000:
                        changes.append(command.tolist())
                    print(f"[sim2sim] cmd={command.tolist()} scale={motion.scale:.2f} paused={motion.paused}", flush=True)
                if not motion.paused:
                    sim.step(command)
                    count += 1
                    min_height = min(min_height, float(sim.data.qpos[2]))
                    max_torque_ratio = max(max_torque_ratio,
                                           float(np.max(np.abs(sim.last_torque) / sim.torque_limits)))
                    if count % sim.decimation == 0:
                        matrix = rotation(sim.data.qpos[3:7])
                        velocity = sim.com_velocity()
                        tilt = math.degrees(math.acos(float(np.clip(matrix[2, 2], -1, 1))))
                        stats += [velocity[0], velocity[1], sim.data.qvel[5], sim.data.qpos[2], tilt]
                        samples += 1
                if viewer is not None and (motion.paused or count % sim.decimation == 0):
                    with viewer.lock():
                        viewer.cam.lookat[:] = sim.data.qpos[:3]
                    viewer.sync()
                if motion.paused:
                    time.sleep(0.02)
                elif not args.fast:
                    time.sleep(max(0, sim.dt - (time.monotonic() - start)))
    except KeyboardInterrupt:
        reason = "interrupt"
    finally:
        if keyboard is not None:
            keyboard.stop()
    result = dict(status="completed", reason=reason, policy=str(policy_path), physics=settings,
                  physics_steps=count, policy_steps=sim.policy_calls, simulation_seconds=count * sim.dt,
                  finite_state=True, min_height=min_height, max_torque_ratio=max_torque_ratio,
                  mean_vx_vy_yaw_height_tilt=(stats / max(samples, 1)).tolist(),
                  command_changes=changes, keyboard_events=event_counts, speed_scale=motion.scale,
                  viewer=not args.headless)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("[sim2sim] RESULT " + json.dumps(result, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    run(parse_args())
