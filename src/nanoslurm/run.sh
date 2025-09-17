#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 -n <job_name> -c <partition> -t <HH:MM:SS> -p <CPUS> -m <GB> [-g <GPUS>] -o <stdout> -e <stderr> -s <SIG@secs> -w <workdir> -- <command...>"
  exit 1
}

# Require sbatch
command -v sbatch >/dev/null 2>&1 || { echo "Error: sbatch not found"; exit 1; }

# Parse with getopts (short flags only)
JOB_NAME=""; PARTITION=""; SBATCH_TIME=""; CPUS=""; MEM_GB=""; GPUS=""
STDOUT_FILE=""; STDERR_FILE=""; SIGNAL=""; WORKDIR=""

while getopts ":n:c:t:p:m:g:o:e:s:w:h" opt; do
  case "$opt" in
    n) JOB_NAME="$OPTARG" ;;
    c) PARTITION="$OPTARG" ;;
    t) SBATCH_TIME="$OPTARG" ;;
    p) CPUS="$OPTARG" ;;
    m) MEM_GB="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    o) STDOUT_FILE="$OPTARG" ;;
    e) STDERR_FILE="$OPTARG" ;;
    s) SIGNAL="$OPTARG" ;;
    w) WORKDIR="$OPTARG" ;;
    h) usage ;;
    \?) echo "Unknown option: -$OPTARG"; usage ;;
    :)  echo "Missing argument for -$OPTARG"; usage ;;
  esac
done
shift $((OPTIND - 1))

# Remaining args form the command to execute
[[ $# -gt 0 ]] || { echo "Error: no command provided"; usage; }
SCRIPT_TO_EXECUTE="$*"

# Enforce required flags (no defaults here)
[[ -n "$JOB_NAME"    ]] || { echo "Error: -n <job_name> is required"; usage; }
[[ -n "$PARTITION"   ]] || { echo "Error: -c <partition> is required"; usage; }
[[ -n "$SBATCH_TIME" ]] || { echo "Error: -t <HH:MM:SS> is required"; usage; }
[[ -n "$CPUS"        ]] || { echo "Error: -p <CPUS> is required"; usage; }
[[ -n "$MEM_GB"      ]] || { echo "Error: -m <GB> is required"; usage; }
[[ -n "$STDOUT_FILE" ]] || { echo "Error: -o <stdout> is required"; usage; }
[[ -n "$STDERR_FILE" ]] || { echo "Error: -e <stderr> is required"; usage; }
[[ -n "$SIGNAL"      ]] || { echo "Error: -s <SIG@secs> is required"; usage; }
[[ -n "$WORKDIR"     ]] || { echo "Error: -w <workdir> is required"; usage; }

TEMP_SCRIPT=$(mktemp /tmp/nanoslurm.XXXXXX)
trap 'rm -f "$TEMP_SCRIPT"' EXIT

# Header (variables expand here as intended)
cat > "$TEMP_SCRIPT" <<EOT
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH -t $SBATCH_TIME
#SBATCH -c $CPUS
#SBATCH --mem=${MEM_GB}G
EOT

if [[ -n "$GPUS" && "$GPUS" =~ ^[0-9]+$ ]]; then
  if (( GPUS > 0 )); then
    echo "#SBATCH --gres=gpu:${GPUS}" >> "$TEMP_SCRIPT"
  fi
fi

cat >> "$TEMP_SCRIPT" <<EOT
#SBATCH --job-name=$JOB_NAME
#SBATCH --signal=$SIGNAL
#SBATCH -o $STDOUT_FILE
#SBATCH -e $STDERR_FILE

set -euo pipefail
EOT

# Append the user command literally (no submit-node expansion)
{
  printf '%s\n' "$SCRIPT_TO_EXECUTE"
} >> "$TEMP_SCRIPT"

chmod +x "$TEMP_SCRIPT"

# Respect caller's working directory at runtime
sbatch -D "$WORKDIR" "$TEMP_SCRIPT"

