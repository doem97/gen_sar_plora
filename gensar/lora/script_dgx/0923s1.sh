#!/bin/bash

# CATEGORIES=("0" "1" "2" "3" "4" "5" "6" "7")
# DATA_SIZE_ARR=("228" "590" "89" "1146" "423" "501" "39" "425")
CATEGORIES=("4" "5" "6" "7")
DATA_SIZE_ARR=("423" "501" "39" "425")

# Parallel execution using a loop
for i in "${!CATEGORIES[@]}"; do
    CAT=${CATEGORIES[$i]}
    DATA_SIZE=${DATA_SIZE_ARR[$i]}
    
    # Define the log file for each script run
    LOG_FILE="./output/cluslora/logs_cluster16/cluster_${CAT}.log"
    
    # Run the training script with the above configurations in the background
    CUDA_VISIBLE_DEVICES="0,1,2,3" CATEGORY=$CAT MAX_STEPS=$((DATA_SIZE*200/32)) CTRL_PORT=29700 bash ./script/cluster/fusrs_clus_rank4.sh > "$LOG_FILE"
    wait
    # Optional: add sleep to stagger the start times if necessary
    sleep 60
done
# Wait for all background processes to complete
wait
