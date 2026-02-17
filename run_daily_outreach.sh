#!/bin/bash

# Configuration
PYTHON_PATH="/Users/niteeshkumar/opt/anaconda3/bin/python3"
PROJECT_DIR="/Users/niteeshkumar/Documents/outreach_ra"
LOG_FILE="$PROJECT_DIR/data/outreach_cron.log"

# Change to project directory
cd "$PROJECT_DIR" || exit

# Run the outreach script
echo "------------------------------------------" >> "$LOG_FILE"
echo "Starting daily outreach at $(date)" >> "$LOG_FILE"
"$PYTHON_PATH" -u ghostwriter/daily_outreach.py --live >> "$LOG_FILE" 2>&1
echo "Completed daily outreach at $(date)" >> "$LOG_FILE"
echo "------------------------------------------" >> "$LOG_FILE"
