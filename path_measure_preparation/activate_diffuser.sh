#!/usr/bin/env bash
B=/root/autodl-tmp/path_measure_preparation
export PATH="$B/env38/bin:$PATH"
export MUJOCO_PY_MUJOCO_PATH="$B/mujoco/mujoco200_linux"
export MUJOCO_PY_MJKEY_PATH="$B/diffuser_7ea4228/azure/files/mjkey.txt"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$B/diffuser_7ea4228:${PYTHONPATH:-}"
