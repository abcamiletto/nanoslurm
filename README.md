# nanoslurm

**nanoslurm** is a zero-dependency Python wrapper for [SLURM](https://slurm.schedmd.com/) job submission and monitoring.  
It uses a tiny POSIX-compatible shell script to call `sbatch` and related commands, avoiding any heavy Python dependencies.

## Features

- **Submit jobs** from Python without `pyslurm` or other packages
- **Monitor status** (`PENDING`, `RUNNING`, `COMPLETED`, etc.)
- **Cancel jobs**
- **Tail job logs**
- **Get detailed info** via `scontrol`
- **Respects working directory** at runtime (`sbatch -D`)

## Requirements

- SLURM cluster with `sbatch`, `squeue`, and optionally `sacct` / `scontrol`
- Python ≥ 3.11
- Linux operating system


## Quickstart

```python
import nanoslurm

job = nanoslurm.submit(
    command=["python", "train.py", "--epochs", "10"],
    name="my_job",
    cluster="gpu22",
    time="01:00:00",
    cpus=4,
    memory=16,
    gpus=1,
    stdout_file="./slurm_logs/%j.txt",
    stderr_file="./slurm_logs/%j.err",
    signal="SIGUSR1@90",
    workdir="."
)

print(job)                      # Job(id=123456, name='my_job_2025-08-08_09-12-33.123', ...)
print(job.status)               # "PENDING", "RUNNING", ...
print(job.is_running())         # True / False
print(job.is_finished())        # True / False
print(job.info())               # Detailed dict from scontrol
job.tail(10)                    # Last 10 lines of stdout
job.wait(poll_interval=5)       # Wait until completion
job.cancel()                    # Cancel job

```

## Command line interface

Install the CLI from PyPI. Use the `nanoslurm` entry point (a shorter
`nslurm` alias is also available):

```bash
pip install nanoslurm
# or, with uv
uv tool install nanoslurm
```

Run `nanoslurm run --help` to see all available options. Use `--` to separate
SLURM options from the command that should run on the cluster. For example:

```bash
nanoslurm run \
  --cluster gpu22 \
  --time 01:00:00 \
  --cpus 4 \
  --memory 16 \
  --gpus 1 \
  --stdout-file ./slurm_logs/%j.txt \
  --stderr-file ./slurm_logs/%j.err \
  -- python train.py --epochs 10
```

Add `--interactive / -i` to be prompted for any values that are not supplied on
the command line:

```bash
nanoslurm run --interactive -- python train.py --epochs 10
```

Persist frequently used defaults (stored as YAML under
`~/.config/nanoslurm/config.yaml`) so you do not have to repeat them:

```bash
nanoslurm defaults show            # list current defaults
nanoslurm defaults set cluster gpu22
nanoslurm defaults set cpus 4
nanoslurm defaults reset
nanoslurm defaults edit            # open the YAML config in $EDITOR
```

Most resource parameters (cluster, time, CPUs, memory, GPUs, etc.) must be set
explicitly or persisted via `nanoslurm defaults set` before submitting a job.

Launch the interactive job monitor to inspect running and pending jobs:

```bash
nanoslurm monitor
```

Use the arrow keys or `h`, `j`, `k`, `l` to move around and `q` to quit.

## Releasing

Bump the version in `pyproject.toml` and merge the change into `main`. A
workflow will tag the commit as `vX.Y.Z` and publish the package to PyPI.
