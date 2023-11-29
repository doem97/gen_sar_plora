#!/bin/bash

RANKS=("1" "2" "8")

# Parallel execution using a loop
for i in "${!RANKS[@]}"; do
    LORA_RANK=${RANKS[$i]}
    # Define the log file for each script run
    LOG_FILE="./output/sarlora/256/logs/rank_${LORA_RANK}.log"
    # Run the training script with the above configurations in the background
    CUDA_VISIBLE_DEVICES="0,1,2,3" LORA_RANK=$LORA_RANK CTRL_PORT=20700 bash ./script/256/train_fusrs_rank_n.sh > "$LOG_FILE"
    wait
    # Optional: add sleep to stagger the start times if necessary
    sleep 60
done
# Wait for all background processes to complete
wait
