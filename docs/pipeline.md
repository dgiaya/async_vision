# AsyncCalibrator — Implementation Status
### File: `/async_vision/Kalibr/kalibr_calibrate_cameras_async.py`

---

## Pipeline Design
```
__init__(cam0Config, cam1Config, targetConfig, dataset0, dataset1)
    --> setup calibration target detector per camera    # setupCalibrationTarget
    --> extract corners for cam0                        # extractCornersFromDataset
    --> extract corners for cam1                        # extractCornersFromDataset
    --> initialize T_cam1_cam0 to identity              # prior, updated externally

initPoseSpline(observations)
    --> fit B-spline through discrete T_t_c poses       # generalized from IccCamera
    --> handles rotation continuity and boundary padding

findTimeshiftPrior()
    --> build splines for both cameras
    --> cross-correlate angular velocity norms          # offset = signal lag
    --> store result in self.timeshiftCam0ToCam1Prior

addDesignVariables(problem, dvc)
    --> register T_cam1_cam0 as TransformationDv        # rotation + translation split
    --> register identity T_cam0_cam0 as inactive DV    # required by CalibrationOptimizationProblem
    --> register temporal offset as aopt.Scalar(0.0)    # residual on top of prior
    --> register spline control points via dvc          # asp.BSplinePoseDesignVariable

addCameraErrorTerms(problem, dataset, camera, observations, T_camN_cam0,
                    poseSplineDv, applyFrameTimeShift)
    --> cam0: frameTime = ScalarExpression(obs.time().toSec())
    --> cam1: frameTime = timeshiftDV + obs.time().toSec() + prior
    --> query spline at frameTime
    --> compute reprojection error with Blake-Zisserman M-estimator

buildAndSolveProblem()
    --> findTimeshiftPrior
    --> initPoseSpline from cam0 observations
    --> wrap as BSplinePoseDesignVariable
    --> addDesignVariables
    --> addCameraErrorTerms for cam0 (identity extrinsic, no time shift)
    --> addCameraErrorTerms for cam1 (T_cam1_cam0 expression, time shift applied)
    --> run Levenberg-Marquardt
    --> return T_cam1_cam0, total time offset

main()
    --> hardcoded paths for testing (parseArgs to be wired later)
    --> load cam0Config, cam1Config, targetConfig
    --> initialize dataset0, dataset1
    --> create AsyncCalibrator
    --> call buildAndSolveProblem
    --> print T_cam1_cam0 and time offset
```

---

## Implementation Status

### Completed

**`__init__`**
- Initializes both cameras independently with separate detectors
- `setupCalibrationTarget` called per camera, self.camera set correctly before each call
- Extracts `target0Observations` and `target1Observations` without sync constraint
- `self.problem = inc.CalibrationOptimizationProblem()`
- `self.T_cam1_cam0 = sm.Transformation()` — identity prior

**`setupCalibrationTarget`**
- Copied from IccCamera, camera passed as explicit argument
- Creates GridDetector tied to camera-specific geometry and intrinsics

**`initPoseSpline(observations)`**
- Generalized from `IccCamera.initPoseSplineFromCamera`
- Uses `obs.T_t_c().T()` — camera pose in target frame, no body frame conversion needed
- Handles rotation vector continuity (prevents 2π flips)
- Handles boundary padding for spline sliding during optimization

**`findTimeshiftPrior`**
- Builds splines for both cameras
- Cross-correlates angular velocity norms at overlapping cam0 timestamps
- Stores result in `self.timeshiftCam0ToCam1Prior`
- Known issue: returns 0.0 on 50ms offset data — under investigation

**`addDesignVariables`**
- `T_cam1_cam0_Dv` registered via `getDesignVariable(i)` 
- `T_cam0_cam0_Dv` registered as inactive — required by `inc.CalibrationOptimizationProblem`
- `cam0TimeTocam1TimeDV = aopt.Scalar(0.0)` registered with `CALIBRATION_GROUP_ID`
- Spline control points registered via `dvc.numDesignVariables()` loop

**`addCameraErrorTerms`**
- Single generalized function for both cameras via `applyFrameTimeShift` flag
- cam0 `frameTime` wrapped as `aopt.ScalarExpression` — required by C++ `transformationAtTime`
- cam1 `frameTime` built as differentiable expression — optimizer can backpropagate
- `T_camN_cam0` passed as expression from caller
- Returns `allReprojectionErrors`

**`buildAndSolveProblem`**
- Full pipeline wired and running end to end
- LM optimizer configured with `BlockCholeskyLinearSystemSolver`
- Returns `sm.Transformation(T_cam1_cam0_Dv.T())` and total time offset

**`main`**
- Hardcoded paths for testing
- Pipeline runs end to end inside Docker

---

### In Progress

**`findTimeshiftPrior` — cross-correlation fix**
- Returns 0.0 for 50ms offset dataset
- Root cause: cam1 angular velocity evaluated at cam0 timestamps without offset compensation
- Signals fed into cross-correlation are misaligned before correlation starts

---

### Not Yet Done

- `parseArgs` wired into `main` — currently hardcoded
- Results saved to yaml output file
- Validation against Kalibr ground truth on synchronized data

---

## Key Technical Notes

- `inc.CalibrationOptimizationProblem` required — `aopt.OptimizationProblem` rejects `TransformationDv` 
  components due to `shared_ptr` mismatch in C++ wrapper
- All DVs referenced by error terms must be registered even if inactive
- Time shift DV initialized to 0.0 — represents residual correction, prior handles bulk offset
- LM is local — if prior is 0.0 instead of 50ms, optimizer cannot recover true offset
- `T_destination_source` convention throughout

---

## Test Results

| Dataset | T_cam1_cam0 baseline | Time shift recovered | Expected |
|---|---|---|---|
| Synchronized | ~0.19m | -2ms | ~0ms |
| 50ms offset (alternateSkipped.bag) | ~0.19m | -4ms | ~50ms |