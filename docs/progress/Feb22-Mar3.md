# Week 9: Feb 24 - Mar 2, 2025

## Overview
Investigated Kalibr camera-IMU pipeline in depth and began implementing async stereo calibration.

---

## Feb 24-28, 2025

### Camera-IMU Pipeline Investigation
Goal: Map reusable patterns from IccSensors.py and IccCalibrator.py for cam-cam adaptation

Findings:
- IccCamera.initPoseSplineFromCamera builds continuous B-spline from discrete PnP poses
- Temporal offset added as aopt.Scalar design variable via addDesignVariables
- Spline queried at t_cam + offset during addCameraErrorTerms
- timeOffsetPadding extends spline range to handle offset sliding during optimization
- noTransformation=False required so obs.T_t_c() pose is attached to each observation
- ObservationDatabase.getAllObsCam(cam_id) returns per-camera observations bypassing sync constraint

Key files mapped:
- IccSensors.py: IccCamera and IccCameraChain classes
- IccCalibrator.py: buildProblem orchestration
- ObsDb.py: observation storage and sync logic

---

## Mar 1-2, 2025

### AsyncCalibrator Implementation
Goal: New file kalibr_calibrate_cameras_async.py adapting cam-IMU approach for cam-cam

Core approach:
- Cam0 discrete poses from AprilGrid PnP form continuous B-spline (reference)
- Cam1 observations query spline at t_cam1 + offset
- Temporal offset jointly optimized with stereo extrinsics T_cam1_cam0

Functions implemented:

__init__
- Initializes both cameras from config independently
- Calls setupCalibrationTarget per camera using camera-specific geometry
- Extracts corners for both cameras without sync constraint

setupCalibrationTarget
- Copied from IccCamera
- Creates GridDetector tied to camera-specific intrinsics
- Must be called after setting self.camera to correct camera

initPoseSpline(observations)
- Generalized from IccCamera.initPoseSplineFromCamera
- Takes any observation list, returns fitted BSplinePose on SE(3)
- No body frame conversion needed, Cam0 is reference so obs.T_t_c().T() used directly
- Handles rotation vector continuity and boundary padding

findTimeshiftPrior
- Builds permanent Cam0 spline and temporary throwaway Cam1 spline
- Evaluates angular velocity norms from both splines at overlapping timestamps
- Cross-correlates signals to estimate initial temporal offset
- Stores result as self.timeshiftCam0ToCam1Prior

---

## Key Insights 

Cam0 is reference so T_c_b is identity, no body frame transformation needed unlike IMU-camera
obs.T_t_c().T() gives T_c_t, camera position relative to fixed target, correct representation for spline
Cam1 temporary spline is throwaway, only used for cross-correlation prior estimation
setupCalibrationTarget uses self.camera.geometry internally so self.camera must point to correct camera before each call
Two detectors needed because GridDetector bakes in camera intrinsics for undistortion and PnP

---

## Weekly Summary

### Completed
Full camera-IMU pipeline mapped and understood
AsyncCalibrator class started with __init__, setupCalibrationTarget, initPoseSpline, findTimeshiftPrior

### In Progress
addDesignVariables
addCam0ErrorTerms and addCam1ErrorTerms
buildAndSolve

### Next Week Goals
1. Complete addDesignVariables with T_cam1_cam0 and temporal offset scalar
2. Implement addCam1ErrorTerms querying spline at t_cam1 + offset
3. Wire full pipeline in buildAndSolve
4. Test on synchronized dataset with known offset as ground truth
