#!/bin/bash
source ./venv/bin/activate
echo "Script started at $(date)" >> experiment.log
python main.py
echo "Script finished at $(date)" >> experiment.log
deactivate
