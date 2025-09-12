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

Install the CLI with [uv](https://github.com/astral-sh/uv):

```bash
uv tool install nanoslurm
```

Submit a job from the terminal:

```bash
nslurm run -c gpu22 -t 01:00:00 -p 4 -m 16 -g 1 -- python train.py --epochs 10
```

Launch an interactive prompt to build a command and adjust options:

```bash
nslurm run -i
```

Manage persistent defaults (stored as YAML via `platformdirs`):

```bash
nslurm defaults show            # list current defaults
nslurm defaults set cluster gpu22
nslurm defaults reset
nslurm defaults edit            # open the YAML config in $EDITOR
```

Most job parameters (such as cluster, time, or resource counts) have no built-in
defaults. Set them explicitly on the command line or persist them via
`nslurm defaults set`.

View current jobs in an interactive TUI:

```bash
nslurm jobs
```

Use the arrow keys or `h`, `j`, `k`, `l` to move around and `q` to quit.

## Releasing

Bump the version in `pyproject.toml` and merge the change into `main`. A
workflow will tag the commit as `vX.Y.Z` and publish the package to PyPI.
