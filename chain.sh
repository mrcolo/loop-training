#!/bin/bash
# Run the pending evaluations, then hand the card back to training and keep it
# there. Nothing in here needs a human between steps.
cd /home/stem-user/loop
# Match the python process only. A shell whose own command line mentions the
# script -- including the one polling for it -- otherwise matches forever.
while pgrep -f "python.*probe_transplant\.py" > /dev/null; do sleep 30; done
echo "=== $(date -Is) transplant probe done, running the evaluation"
.venv/bin/python -u evaluate.py --write > eval.log 2>&1
echo "=== $(date -Is) evaluation exit $?, resuming training"
exec ./supervise.sh
