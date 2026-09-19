#!/bin/bash
N=${1:-10}
CARLA_DIR=${CARLA_DIR:-$HOME/CARLA_0.9.15}

echo "recording $N scenario(s), one CARLA session each"
echo

for i in $(seq 0 $((N-1))); do
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
    echo "scenario $i: exceeded 30 minutes, moving on"
  else
    echo "scenario $i: failed with status $status, moving on"
  fi
  echo
done

pkill -f CarlaUE4 2>/dev/null
echo "=============================================="
echo "done. recorded files:"
ls -1 data/attack_trajectories/*__attacked.json 2>/dev/null | wc -l
ls -1 data/attack_trajectories/ 2>/dev/null | sed 's/__.*//' | sort -u