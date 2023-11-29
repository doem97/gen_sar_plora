#!/bin/bash

CATEGORIES=("2" "3")
DATA_SIZE_ARR=("569" "2690")

# Parallel execution using a loop
for i in "${!CATEGORIES[@]}"; do
    CAT=${CATEGORIES[$i]}
    DATA_SIZE=${DATA_SIZE_ARR[$i]}
    
    # Define the log file for each script run
    LOG_FILE="./output/cluslora/logs_cluster4/cluster_${CAT}.log"
    
    # Run the training script with the above configurations in the background
    CUDA_VISIBLE_DEVICES="4,5,6,7" CATEGORY=$CAT MAX_STEPS=$((DATA_SIZE*200/32)) CTRL_PORT=31700 bash ./script/cluster/fusrs_clus_rank4_c4.sh > "$LOG_FILE"
    wait
    # Optional: add sleep to stagger the start times if necessary
    sleep 60
done
# Wait for all background processes to complete
wait