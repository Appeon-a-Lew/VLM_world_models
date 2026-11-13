"""Resume an audited run after data generation, preserving failed startup artifacts.

Launch with the training venv and a six-hour outer timeout. Uses the frozen code/ snapshot.
"""
import ctypes
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
os.chdir(root / 'code')
sys.path.insert(0, str(root / 'code'))
os.environ.update(PYTHONPATH=str(root / 'code'), HF_HOME='/var/tmp/toyi/wm/hf',
                  XDG_CACHE_HOME='/var/tmp/toyi/wm/cache', MUJOCO_GL='egl',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                  TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
lock = (root / 'run.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
stage = 'recovery_validation'


def status(state, detail=''):
    rec = dict(state=state, stage=stage, detail=detail, host=socket.gethostname(), pid=os.getpid(),
               updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (root / 'status.tmp').write_text(json.dumps(rec, indent=2) + '\n')
    (root / 'status.tmp').replace(root / 'status.json')
    with (root / 'events.jsonl').open('a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def wait_gpu():
    while True:
        rc = ctypes.CDLL('libcuda.so.1').cuInit(0)
        if rc != 0:
            raise RuntimeError(f'CUDA driver initialization failed: {rc}')
        values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits'], text=True)
        mem, util = [x.strip() for x in values.splitlines()[0].split(',')]
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True)
        if int(mem) < 2500 and util.isdigit() and int(util) <= 5 and not any(x.strip().isdigit() for x in apps.splitlines()):
            status('running')
            return
        status('waiting_for_gpu', 'Healthy GPU occupied; retrying in 60 seconds')
        time.sleep(60)


def run(name, args, gpu=True, timeout=10800):
    global stage
    stage = name
    if gpu:
        wait_gpu()
    status('running')
    with (root / 'logs' / f'{name}.log').open('x') as log:
        subprocess.run([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


try:
    import hashlib
    from physwm.schema import load_split

    metadata = json.loads((root / 'launch.json').read_text())
    archive = root / 'physwm-audit-v2-20261027.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata['archive_sha256']
    assert (root / 'recovery_copy_complete').exists()
    splits = ['val_id', 'val_ood_objects', 'val_ood_forces', 'val_ood_config',
              'val_diag_elev', 'val_diag_count', 'val_diag_corner', 'val_diag_elev_corner']
    for split in ['train', *splits]:
        expected = 10000 if split == 'train' else 300 if split.startswith('val_diag') else 500
        assert len(load_split(root / 'data', split)) == expected, split
    assert not (root / 'model' / 'adapter').exists(), 'Do not overwrite a trained adapter'
    if (root / 'model').exists():
        (root / 'model').rename(root / 'model_failed_cuda')
    run('train_retry', ['-m', 'physwm.vlm.sft', '--config', str(root / 'sft_config.yaml')])
    common = ['-m', 'physwm.vlm.infer', '--data', str(root / 'data'), '--lora',
              str(root / 'model/adapter'), '--batch-size', '32']
    run('infer_normal_retry', [*common, '--splits', *splits, '--out', str(root / 'pred/normal.jsonl')])
    for mode in ['none', 'blank', 'swap']:
        run(f'infer_{mode}_retry', [*common, '--splits', 'val_id', '--image-mode', mode,
                                  '--out', str(root / f'pred/{mode}.jsonl')])
    run('evaluate_retry', ['-m', 'physwm.eval.evaluate', '--data', str(root / 'data'),
        '--splits', *splits, '--csv', str(root / 'results/metrics.csv'), '--pred',
        f'static={root}/baselines/static.jsonl', f'const_velocity={root}/baselines/const_velocity.jsonl',
        *[f'sft_{mode}={root}/pred/{mode}.jsonl' for mode in ['normal', 'none', 'blank', 'swap']]], gpu=False)
    (root / 'results/table.md').write_text((root / 'logs/evaluate_retry.log').read_text())
    stage = 'complete'
    status('done', 'Corrected validation reference and ablations finished')
except BaseException as exc:
    status('failed', f'{type(exc).__name__}: {exc}')
    raise
