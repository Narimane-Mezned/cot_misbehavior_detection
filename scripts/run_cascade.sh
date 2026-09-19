#!/bin/bash
set -u
mkdir -p outputs/logs/cascade
LOG=outputs/logs/cascade

run() {
  name=$1; shift
  echo "===================================================================="
  echo "  $name"
  echo "===================================================================="
  if python3 -u "$@" > "$LOG/$name.log" 2>&1; then
    echo "  OK    -> $LOG/$name.log"
    tail -5 "$LOG/$name.log" | sed 's/^/        /'
  else
    echo "  FAILED -> $LOG/$name.log"
    tail -15 "$LOG/$name.log" | sed 's/^/        /'
  fi
  echo
}

echo "STEP 1 -- calibration (everything below depends on this)"
run calibrate scripts/calibrate_detector.py

echo "STEP 2 -- adversarial ML attacks"
run ml_attacks      scripts/evaluate_ml_attacks_real_data.py
run db_recon        scripts/evaluate_database_reconstruction_trials.py

echo "STEP 3 -- backdoor defence"
run backdoor        scripts/evaluate_backdoor_defense.py
run variance        scripts/evaluate_variance.py
run threat_spectrum scripts/evaluate_threat_model_spectrum.py
run calib_methods   scripts/evaluate_calibration_methods.py
run false_positive  scripts/evaluate_false_positive_cost.py
run decoupled       scripts/evaluate_decoupled_scale.py
run local_sweep     scripts/evaluate_local_defense_sweep.py

echo "STEP 4 -- feature analysis"
run attribute_mit   scripts/evaluate_attribute_mitigation.py
run robustness      scripts/evaluate_robustness.py
run feature_dom     scripts/diagnose_feature_dominance.py

echo "STEP 5 -- captions"
run cot_dataset     scripts/build_cot_dataset.py
run cot_poisoning   scripts/build_cot_dataset_poisoning_attacks.py
run cot_quality     scripts/evaluate_cot_quality.py

echo "===================================================================="
echo "  cascade finished"
echo "===================================================================="
grep -l "Traceback" $LOG/*.log 2>/dev/null && echo "  ^ these failed" || echo "  no failures"