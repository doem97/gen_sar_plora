#!/bin/bash
# @doem97: 
# Because Linux Bash doesn't support edit running script on the fly, 
# so I use this work around to cache bash scripts.

script=$(<./scripts/inf_scripts.sh) # Read the contents of script.sh into a variable
eval "$script" # Execute the contents of the variable
