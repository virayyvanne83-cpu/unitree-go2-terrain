"""Behavioral checks for the supplied 48-D controller; no optimizer or training."""
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('go2_controller', ROOT/'deploy/deploy_mujoco/deploy_go2.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def main():
    cfg = yaml.safe_load((ROOT/'deploy/deploy_mujoco/configs/go2.yaml').read_text())
    keyboard = controller.KeyboardState()
    motion = controller.MotionCommands(cfg)
    keyboard.press('w'); keys, events = keyboard.snapshot()
    np.testing.assert_allclose(motion.update(keys,events), [0.4,0,0])
    keyboard.release('w'); keys, events = keyboard.snapshot()
    np.testing.assert_allclose(motion.update(keys,events), [0,0,0])
    keyboard.press('w'); keyboard.press('space'); keys, events = keyboard.snapshot()
    np.testing.assert_allclose(motion.update(keys,events), [0,0,0])
    keyboard.release('space'); keys, events = keyboard.snapshot()
    np.testing.assert_allclose(motion.update(keys,events), [0,0,0])
    keyboard.release('w'); keyboard.press('w'); keys, events = keyboard.snapshot()
    np.testing.assert_allclose(motion.update(keys,events), [0.4,0,0])
    motion.update(set(),[']']*100); assert motion.scale == 1.5
    motion.update(set(),['[']*100); assert motion.scale == 0.1
    motion.update(set(),['p']); assert motion.paused
    motion.update(set(),['p']); assert not motion.paused
    # Verify the desktop hook can start and close without sending OS key events.
    native_keyboard = controller.KeyboardState()
    try:
        native_keyboard.start()
        assert native_keyboard.listener.is_alive()
    finally:
        native_keyboard.stop()

    simulation = controller.Go2Simulation(cfg, controller.resolve(cfg['policy_path']))
    np.testing.assert_allclose(simulation.torque_limits, [23.7,23.7,35.55]*4)
    assert simulation.model.nq == 19 and simulation.model.nv == 18
    simulation.data.qpos[3:7] = [np.cos(0.3),0,0,np.sin(0.3)]
    simulation.data.qvel[:6] = [1.2,-0.3,0.05,0.2,-0.1,0.6]
    mujoco.mj_forward(simulation.model,simulation.data)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(simulation.model, simulation.data, mujoco.mjtObj.mjOBJ_BODY,
                            simulation.base_id, velocity, 0)
    # Native global COM velocity -> body axes, independent of the inertial frame orientation.
    body_com_velocity = controller.rotation(simulation.data.qpos[3:7]).T @ velocity[3:]
    np.testing.assert_allclose(simulation.com_velocity(), body_com_velocity, atol=1e-10)
    observation = simulation.observe(np.array([0.4,0,0]))
    assert observation.shape == (48,) and np.isfinite(observation).all()
    np.testing.assert_allclose(observation[:3],body_com_velocity*2.,atol=1e-6)
    np.testing.assert_allclose(observation[9:12],[0.8,0,0])
    np.testing.assert_allclose(observation[36:48],0.)
    simulation.reset()
    simulation.target[:] = simulation.default_angles + 100.
    simulation.control_steps = 1  # Hold a deliberately large PD target for a non-policy tick.
    simulation.step(np.zeros(3))
    assert np.all(np.abs(simulation.last_torque) <= simulation.torque_limits)
    np.testing.assert_allclose(simulation.last_torque,simulation.torque_limits)
    simulation.reset()
    for _ in range(40):
        simulation.step(np.array([0.4,0,0]))
    assert simulation.policy_calls == 10
    assert np.isfinite(simulation.data.qpos).all()
    simulation.reset()
    np.testing.assert_allclose(simulation.data.qpos[simulation.qidx],simulation.default_angles)
    np.testing.assert_allclose(simulation.action,0.)
    report = dict(passed=True, weights_updated=False, checks=[
        'press/release command', 'space blocks held keys until repress', 'speed bounds',
        'pause toggle', 'native X11 keyboard hook lifecycle', '48-D observation slices',
        'COM velocity agrees with native mj_objectVelocity', '23.7/35.55 torque saturation',
        '50 Hz policy timing', 'finite MuJoCo state', 'reset clears actions and joints'])
    destination=ROOT/'verification/source_alignment/controller_checks.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
