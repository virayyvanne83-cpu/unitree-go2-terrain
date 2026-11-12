# Go2 MuJoCo model provenance

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_go2

The MJCF, scene, assets and BSD 3-Clause license come from this public reference. The only robot XML adaptation is naming the existing free joint `root`, as required by the supplied 48-D controller. Inertias and joint/contact defaults are retained. The controller applies the supplied 23.7 / 35.55 N m torque limits and disables robot self-collisions while retaining ground contact.

Compiled mass: 15.206408 kg. Original CLT/Lain XML/assets were not supplied, so matching mass does not establish per-body inertia, joint damping or contact equivalence. The legacy `../mjcf` path is a symlink to this directory for earlier generated assets.
