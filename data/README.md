# DeepAccident data notes

## Source

- Mini sample (20 scenarios, 9.0 GB): Google Drive, via `scripts/download_deepaccident_mini.py`
- Full dataset (691 scenarios, ~313 GB, 10 train + 2 val + 2 test zips): see full download script (Phase 1.1, later)

## Structure (per documented file layout — TO BE CONFIRMED against actual mini sample)

```
DeepAccident_data/
  <type>_<subtype>_accident/
    ego_vehicle/
      BEV_instance_camera/<scenario>/<scenario>_<frame>.npz
      calib/<scenario>/<scenario>_<frame>.pkl
      Camera_Back/<scenario>/<scenario>_<frame>.jpg
      Camera_BackLeft/...
      Camera_BackRight/...
      Camera_Front/...
      Camera_FrontLeft/...
      Camera_FrontRight/...
      label/<scenario>/<scenario>_<frame>.txt
      lidar01/<scenario>/<scenario>_<frame>.npz
    ego_vehicle_behind/     (same sub-structure)
    infrastructure/         (same sub-structure)
    other_vehicle/          (same sub-structure)
    other_vehicle_behind/   (same sub-structure)
    meta/<scenario>_<frame>.txt
  <type>_<subtype>_normal/  (same sub-structure, non-accident scenarios)
```

- 5 agents per scenario: `ego_vehicle`, `ego_vehicle_behind`, `infrastructure`, `other_vehicle`, `other_vehicle_behind`
- Each agent: 6 RGB cameras + LiDAR + calibration + labels
- `meta/` holds scenario-level metadata (one level up from per-agent folders)

## Verification log

- [ ] Mini sample downloaded and extracted
- [ ] Top-level scenario-type folders match expected naming (`type#_subtype#_accident` / `_normal`)
- [ ] Per-agent folder structure matches above
- [ ] `label/*.txt` format inspected — confirm what fields it contains (used for Phase 1.2 attack ground truth and Phase 3.2 scene features)
- [ ] `meta/*.txt` format inspected — confirm scenario-level fields (weather, town, accident type)
- [ ] 12-way accident-subtype category balance checked empirically (per plan Section 1.1 — NOT assumed from the 26.9/73.1 split alone)
