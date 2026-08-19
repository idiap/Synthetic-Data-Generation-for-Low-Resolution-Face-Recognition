#!/bin/bash
#
# SPDX-FileCopyrightText: © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Luis S. Luevano <luis.luevano@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Script: monitor_and_relaunch.sh
#
# Monitor and auto-relaunch SLURM training jobs
#
# Usage: ./slurm/monitor_and_relaunch.sh [MONITOR_OPTS] <sbatch_script> [SCRIPT_ARGS...]
#
# Monitor options (must come BEFORE the script name):
#   -n, --max-relaunches N   Max number of job relaunches (default: 5, i.e. 5*8h = 40h)
#   --from-checkpoint        Pass --resume-override to the script on the very first launch.
#                            (All subsequent relaunches always pass --resume-override automatically.)
#
# Everything after the script name is forwarded verbatim to sbatch as script arguments.
# This means flags like --resume-override that belong to the .run script go AFTER the script name.
#
# Examples:
#   ./slurm/monitor_and_relaunch.sh slurm/train_edgeface_lr.run area cubic 56
#   ./slurm/monitor_and_relaunch.sh -n 6 slurm/train_edgeface_lr.run area cubic 56
#   ./slurm/monitor_and_relaunch.sh -n 6 --from-checkpoint slurm/train_edgeface_lr.run area cubic 56
#   ./slurm/monitor_and_relaunch.sh slurm/train_edgeface_lr.run area cubic 56 --resume-override  # forwarded to sbatch

MAX_RELAUNCHES=5
START_WITH_RESUME="false"
SBATCH_SCRIPT=""
SBATCH_EXTRA_ARGS=()

# Parse monitor options; stop at the first non-option argument (the script name)
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-relaunches|-n)
            MAX_RELAUNCHES="$2"
            shift 2
            ;;
        --from-checkpoint)
            START_WITH_RESUME="true"
            shift
            ;;
        *)
            SBATCH_SCRIPT="$1"
            shift
            SBATCH_EXTRA_ARGS=("$@")  # everything remaining goes to sbatch
            break
            ;;
    esac
done

if [ -z "$SBATCH_SCRIPT" ]; then
    echo "Usage: $0 [--max-relaunches N] [--from-checkpoint] <sbatch_script> [script_args...]"
    echo "Example: $0 slurm/train_edgeface_lr.run cubic area 56"
    echo "         $0 -n 5 slurm/train_edgeface_lr.run cubic area 56"
    echo "         $0 slurm/train_edgeface_lr.run area cubic 56"
    echo "         $0 -n 6 slurm/train_edgeface_lr.run area cubic 56"
    echo "         $0 -n 6 --from-checkpoint slurm/train_edgeface_lr.run area cubic 56"
    exit 1
fi

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "Error: SBATCH script '$SBATCH_SCRIPT' not found!"
    exit 1
fi

# Create unique state files based on script path + extra args to avoid conflicts
SCRIPT_HASH=$(echo "$SBATCH_SCRIPT ${SBATCH_EXTRA_ARGS[*]}" | md5sum | cut -d' ' -f1 | cut -c1-8)
SCRIPT_BASENAME=$(basename "$SBATCH_SCRIPT" .run)
STATE_DIR=".monitor_states"
mkdir -p "$STATE_DIR"

STATE_FILE="${STATE_DIR}/state_${SCRIPT_BASENAME}_${SCRIPT_HASH}"
RESUME_FLAG_FILE="${STATE_DIR}/resume_${SCRIPT_BASENAME}_${SCRIPT_HASH}"
JOB_ID_FILE="${STATE_DIR}/jobid_${SCRIPT_BASENAME}_${SCRIPT_HASH}"

# Only resume monitoring if a job from a previous session is still running in SLURM.
# Otherwise the state files are stale (previous monitor was killed / job ended) and
# would incorrectly force --resume-override on a fresh invocation.
RESUME_MONITORING="false"
if [ -f "$STATE_FILE" ] && [ -f "$JOB_ID_FILE" ]; then
    PREV_JOB_ID=$(cat "$JOB_ID_FILE")
    if [ -n "$PREV_JOB_ID" ] && squeue -j "$PREV_JOB_ID" &>/dev/null; then
        RESUME_MONITORING="true"
    fi
fi

if [ "$RESUME_MONITORING" = "false" ]; then
    # Wipe any stale state from a prior interrupted session
    rm -f "$STATE_FILE" "$RESUME_FLAG_FILE" "$JOB_ID_FILE"
    echo "0" > "$STATE_FILE"
    echo "$START_WITH_RESUME" > "$RESUME_FLAG_FILE"
    echo "=== Starting new training session ==="
    echo "Script: $SBATCH_SCRIPT"
    echo "Max relaunches: $MAX_RELAUNCHES"
    echo "Start from checkpoint: $START_WITH_RESUME"
    echo "State files: $STATE_DIR/*_${SCRIPT_BASENAME}_${SCRIPT_HASH}"
else
    RELAUNCH_COUNT=$(cat "$STATE_FILE")
    echo "=== Resuming monitoring of active job $PREV_JOB_ID (relaunch count: $RELAUNCH_COUNT) ==="
    echo "State files: $STATE_DIR/*_${SCRIPT_BASENAME}_${SCRIPT_HASH}"
fi

# Build dynamic job label from script name + extra args (e.g. "train_edgeface_lr_area_cubic_56")
build_job_opts() {
    local label="${SCRIPT_BASENAME}"
    if [ ${#SBATCH_EXTRA_ARGS[@]} -gt 0 ]; then
        label="${label}_$(IFS=_; echo "${SBATCH_EXTRA_ARGS[*]}")"
    fi
    JOB_OPTS=(
        --job-name="${label}"
        --output="logs/${label}_%A_%a.out"
        --error="logs/${label}_%A_%a.err"
    )
}

# Function to submit job with or without resume flag
submit_job() {
    local use_resume=$1
    local relaunch_num=$2

    build_job_opts

    echo ""
    echo "=== Launch #$relaunch_num at $(date) ==="

    # Build the final args list, adding --resume-override when needed.
    # The .run script forwards --resume-override directly to the Python training script.
    local extra_args=("${SBATCH_EXTRA_ARGS[@]}")
    if [ "$use_resume" = "true" ]; then
        echo "Resuming from checkpoint (passing --resume-override to script)..."
        if [[ ! " ${extra_args[*]} " =~ " --resume-override " ]]; then
            extra_args+=("--resume-override")
        fi
    else
        echo "Launching initial training..."
    fi

    JOB_ID=$(sbatch --parsable "${JOB_OPTS[@]}" "$SBATCH_SCRIPT" "${extra_args[@]}")
    
    if [ -z "$JOB_ID" ]; then
        echo "ERROR: Failed to submit job!"
        return 1
    fi
    
    echo "Job submitted: $JOB_ID"
    echo "$JOB_ID" > "$JOB_ID_FILE"
    return 0
}

# Function to check job status
check_job_status() {
    local job_id=$1
    squeue -j "$job_id" &>/dev/null
    return $?
}

# Main monitoring loop
RELAUNCH_COUNT=$(cat "$STATE_FILE")
USE_RESUME=$(cat "$RESUME_FLAG_FILE")

# Submit initial or resumed job
if ! submit_job "$USE_RESUME" $((RELAUNCH_COUNT + 1)); then
    echo "Failed to submit job. Exiting."
    exit 1
fi

CURRENT_JOB_ID=$JOB_ID

# Monitor the job
echo "Monitoring job $CURRENT_JOB_ID..."
echo "Press Ctrl+C to stop monitoring (job will continue running)"
echo ""

while true; do
    if check_job_status "$CURRENT_JOB_ID"; then
        # Job is still running
        sleep 60  # Check every minute
    else
        # Job finished
        echo ""
        echo "=== Job $CURRENT_JOB_ID finished at $(date) ==="
        
        # Increment relaunch counter
        RELAUNCH_COUNT=$((RELAUNCH_COUNT + 1))
        echo "$RELAUNCH_COUNT" > "$STATE_FILE"
        
        # Check if we should relaunch
        if [ $RELAUNCH_COUNT -lt $MAX_RELAUNCHES ]; then
            echo "Relaunch $RELAUNCH_COUNT of $MAX_RELAUNCHES"
            echo "Waiting 30 seconds before relaunching..."
            sleep 30
            
            # Enable resume for all subsequent launches
            echo "true" > "$RESUME_FLAG_FILE"
            
            # Relaunch
            if ! submit_job "true" $((RELAUNCH_COUNT + 1)); then
                echo "Failed to relaunch job. Exiting."
                exit 1
            fi
            
            CURRENT_JOB_ID=$JOB_ID
            echo "Monitoring job $CURRENT_JOB_ID..."
        else
            echo ""
            echo "=== Training complete! ==="
            echo "Reached maximum relaunches ($MAX_RELAUNCHES)"
            echo "Total runtime: approximately $((MAX_RELAUNCHES * 8)) hours"
            
            # Save final job ID before cleaning up
            FINAL_JOB=$CURRENT_JOB_ID
            
            # Clean up state files
            echo "Cleaning up state files..."
            rm -f "$STATE_FILE" "$RESUME_FLAG_FILE" "$JOB_ID_FILE"
            
            echo "Final job ID: $FINAL_JOB"
            exit 0
        fi
    fi
done
