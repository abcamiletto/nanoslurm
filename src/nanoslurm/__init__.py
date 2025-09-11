import sys

if not sys.platform.startswith("linux"):
    raise OSError("nanoslurm is only supported on Linux")


def main() -> None:
    print("Hello from slurmkit!")
