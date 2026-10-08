#!/bin/bash
# usage: run_all_scenarios.sh [END] [START]
#   run_all_scenarios.sh 7        scenarios 0..6   (the original behaviour)
#   run_all_scenarios.sh 20 7     scenarios 7..19  (continue without redoing
#                                                   work already recorded)
END=${1:-10}
START=${2:-0}
CARLA_DIR=${CARLA_DIR:-$HOME/CARLA_0.9.15}

rm -f /tmp/failed_scenarios.txt
echo "recording scenario indices $START..$((END-1)), one CARLA session each"
echo

for i in $(seq "$START" $((END-1))); do
  echo "=============================================="
  echo "scenario index $i"
  echo "=============================================="

  pkill -f CarlaUE4 2>/dev/null
  sleep 5

  "$CARLA_DIR/CarlaUE4.sh" -RenderOffScreen -quality-level=Low > /tmp/carla_$i.log 2>&1 &
  sleep 25

  timeout 1800 python scripts/record_attack_trajectories.py --scenario_index "$i" --resume
  status=$?

  if [ $status -eq 0 ]; then
    echo "scenario $i: completed"
  elif [ $status -eq 124 ]; then
    echo "scenario $i: FAILED -- exceeded 30 minutes, moving on"
    echo "$i" >> /tmp/failed_scenarios.txt
  else
    echo "scenario $i: FAILED -- see the log above, moving on"
    echo "$i" >> /tmp/failed_scenarios.txt
  fi
  echo
done

pkill -f CarlaUE4 2>/dev/null
echo "=============================================="
echo "done. recorded files:"
ls -1 data/attack_trajectories/*__attacked.json 2>/dev/null | wc -l
ls -1 data/attack_trajectories/ 2>/dev/null | sed 's/__.*//' | sort -u
if [ -f /tmp/failed_scenarios.txt ]; then
  echo
  echo "scenario indices that FAILED and produced nothing:"
  cat /tmp/failed_scenarios.txt
fi