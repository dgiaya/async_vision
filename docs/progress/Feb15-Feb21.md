# Week 8: Feb 17-23, 2025

## Overview
Set up development environment for Kalibr modifications and built trajectory analysis tools.

---

## Feb 22, 2025 - Saturday

### Docker Environment Setup
**Goal:** Enable local editing of Kalibr source code

**Actions:**
- Extracted Kalibr source from container `quirky_morse`
- Location: `~/Kalibr/src/kalibr/`
- Initialized Git: `git init && git commit -m "Initial"`
- Created branch: `cam-timesync`

**Aliases created:**
```bash
alias kalibr_dev='xhost +local:root && docker run -it \
  -e "DISPLAY" \
  -e "QT_X11_NO_MITSHM=1" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v "/home/sid/Kalibr/src/kalibr:/catkin_ws/src/kalibr:rw" \
  -v "/home/sid/NeuROAM_data:/data:rw" \
  -v "/home/sid/Kalibr:/output:rw" \
  --name kalibr_dev \
  kalibr-modified \
  /bin/bash'
```

**Result:** Can now edit Kalibr source with VS Code on host

---

### Trajectory Visualization Tools
**Goal:** Visualize camera trajectories from Kalibr exports

**Implementation:**
- Class: `SE3BSplineTrajectory`
- Features: B-spline fitting, Slerp interpolation, matplotlib visualization
- Location: `~/async_vision/scripts/cam_traj_matplot.py`

**Validation:**
- Input: 96 poses over 67 seconds
- Position RMSE: 0.0000 mm
- Rotation RMSE: 0.000000°

**Result:**  Sub-pixel interpolation accuracy validated

---

## Feb 23, 2025 - Sunday

### Kalibr Source Code Investigation
**Goal:** Understand temporal synchronization implementation

**Findings:**
- Temporal constraint in `ObsDb.py` function `addObservation()`
- Hard threshold: `max_delta_approxsync` (typically 20ms)
- With 30ms offset: graph becomes disconnected

**Key insight:** Camera-IMU uses continuous trajectory query, not discrete frame matching

**Next:** Investigate camera-IMU code to find reusable patterns

---

## Weekly Summary

### Completed
 Development environment with Docker volume mounts  
 Git version control for Kalibr modifications  
 Continuous-time trajectory reconstruction tools  
 Identified root cause of temporal sync failure  

### In Progress
 Investigating camera-IMU temporal offset implementation  
 Locating B-spline classes in Schweizer-Messer  

### Blockers
None

### Next Week Goals
1. Map camera-IMU code patterns
2. Quick validation: remove time constraint
3. Quantify calibration degradation with temporal offset