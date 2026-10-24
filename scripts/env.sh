# Source on a lab machine:  source ~/adl4cv-worlmodels/scripts/env.sh
export WM=/var/tmp/toyi/wm                  # machine-local: venv, data, HF cache, runs
export HF_HOME=$WM/hf
export PIP_CACHE_DIR=$WM/cache XDG_CACHE_HOME=$WM/cache
export MUJOCO_GL=egl
export PYTHONPATH=$HOME/adl4cv-worlmodels${PYTHONPATH:+:$PYTHONPATH}
source $WM/venv/bin/activate
cd $HOME/adl4cv-worlmodels
