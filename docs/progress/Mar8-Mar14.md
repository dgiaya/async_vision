# Week: March 8 - March 14, 2025

## Mar 3-6, 2025

### Optimization Setup
Goal: Wire design variables and error terms into optimization problem

Functions implemented:

addDesignVariables
- Registers T_cam1_cam0 as TransformationDv (rotation + translation split for SO(3) manifold)
- Registers temporal offset as aopt.Scalar DV initialized to 0.0 (residual correction on top of prior)
- Registers cam0 spline control points via asp.BSplinePoseDesignVariable
- Registers identity T_cam0_cam0 as inactive DV — required by inc.CalibrationOptimizationProblem even if inactive

addCameraErrorTerms (generalized for both cameras)
- Single function handles both cam0 and cam1 via applyFrameTimeShift flag
- cam0: frameTime = aopt.ScalarExpression(obs.time().toSec()) — plain float must be wrapped for C++ transformationAtTime
- cam1: frameTime = timeshiftDV.toExpression() + obs.time().toSec() + prior — differentiable expression
- T_camN_cam0 passed as expression: identity expression for cam0, T_cam1_cam0_Dv.toExpression() for cam1
- Blake-Zisserman M-estimator support for outlier robustness

buildAndSolveProblem
- Orchestrates full pipeline: prior estimation → spline construction → DV registration → error terms → LM solve
- Uses inc.CalibrationOptimizationProblem (required — aopt.OptimizationProblem rejects TransformationDv components)
- Returns T_cam1_cam0 and total time offset = DV.toScalar() + timeshiftCam0ToCam1Prior

---

## Mar 7-9, 2025

### Pipeline Validation and Debugging

Key bugs resolved during integration:
- inc.CalibrationOptimizationProblem vs aopt.OptimizationProblem — latter rejects TransformationDv via shared_ptr mismatch
- transformationAtTime requires ScalarExpression not plain float — cam0 frameTime wrapped accordingly
- T_cam0_cam0_Dv must be registered with problem despite being inactive — framework enforces all DVs in expression graph are known

Test results on synchronized data:
- T_cam1_cam0 baseline ~0.19m, rotation plausible
- Time shift ~-2ms (consistent with near-synchronized data)

Test results on 50ms artificially offset data:
- Pipeline runs but recovers only -4ms
- Root cause: findTimeshiftPrior returns 0.0 — cross-correlation fails to detect offset

## Key Insights 

- inc.CalibrationOptimizationProblem is a strict superset of aopt.OptimizationProblem — all DVs in expression graph must be registered
- TransformationDv is a pure Python wrapper — underlying C++ DVs are .q (RotationQuaternionDv) and .t (EuclideanPointDv)
- Time shift DV initialized to 0.0 is correct — represents residual correction, prior handles bulk offset
- LM is a local optimizer — if prior is wrong (0.0 instead of 50ms), optimizer cannot recover true offset

---
