#!/usr/bin/env bash
set -euo pipefail
B=/root/autodl-tmp/path_measure_preparation
export MUJOCO_PY_MUJOCO_PATH=$B/mujoco/mujoco200_linux
export MUJOCO_PY_MJKEY_PATH=$B/diffuser_7ea4228/azure/files/mjkey.txt
export LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}
P=$B/env38/bin/python
"$P" -m pip install --no-index --no-deps --no-build-isolation "$B/wheels38/mujoco-py-2.0.2.13.tar.gz"
"$P" -m pip install --no-index --no-deps --no-build-isolation "$B/d4rl_f2a05c0"
"$P" -c "import mujoco_py, gym, d4rl; print('imports OK'); env=gym.make('hopper-medium-v2'); print('reset',env.reset().shape); print('step',env.step(env.action_space.sample())[1])"
echo MUJOCO_CHECK_PASSED
