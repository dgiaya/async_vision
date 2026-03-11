import kalibr_common as kc
import aslam_cv as acv
import aslam_cameras_april as acv_april
import aslam_backend as aopt
import aslam_cv_backend as acvb
import aslam_splines as asp
import incremental_calibration as inc
import numpy as np
import sm
import sys
import bsplines
import signal
import argparse 
import math
import pylab as pl
import multiprocessing

CALIBRATION_GROUP_ID = 0
HELPER_GROUP_ID = 1

#read image topics from the dataset
def initBagDataset(bagfile, topic, from_to, freq):
    print("\tDataset:          {0}".format(bagfile))
    print("\tTopic:            {0}".format(topic))
    reader = kc.BagImageDatasetReader(bagfile, topic, bag_from_to=from_to, bag_freq=freq)
    print("\tNumber of images in the bag: {0}".format(reader.numImages()))
    return reader

#available models
cameraModels = { 'pinhole-radtan': acvb.DistortedPinhole,
                 'pinhole-equi':   acvb.EquidistantPinhole,
                 'pinhole-fov':    acvb.FovPinhole,
                 'omni-none':      acvb.Omni,
                 'omni-radtan':    acvb.DistortedOmni,
                 'eucm-none':      acvb.ExtendedUnified,
                 'ds-none':        acvb.DoubleSphere}

#for interupting pipeline 
def signal_exit(signal, frame):
    sm.logWarn("Shutdown requested! (CTRL+C)")
    sys.exit(2)

def parseArgs():
    class KalibrArgParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            sm.logError('%s' % message)
            sys.exit(2)
        def format_help(self):
            formatter = self._get_formatter()
            formatter.add_text(self.description)
            formatter.add_usage(self.usage, self._actions,
                                self._mutually_exclusive_groups)
            for action_group in self._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            formatter.add_text(self.epilog)
            return formatter.format_help()     
        
    usage = """
    Example usage to calibrate a camera system with two cameras using an aprilgrid. 
    
    cam0: omnidirection model with radial-tangential distortion
    cam1: pinhole model with equidistant distortion
    
    %(prog)s --models omni-radtan pinhole-equi --target aprilgrid.yaml \\
              --bag MYROSBAG.bag --topics /cam0/image_raw /cam1/image_raw
    
    example aprilgrid.yaml:
        target_type: 'aprilgrid'
        tagCols: 6
        tagRows: 6
        tagSize: 0.088  #m
        tagSpacing: 0.3 #percent of tagSize"""
            
    parser = KalibrArgParser(description='Calibrate the intrinsics, extrinsics, time offset of a camera system with non-shared overlapping field of view.', usage=usage)
    parser.add_argument('--models', nargs='+', dest='models', help='The camera model {0} to estimate'.format(list(cameraModels.keys())), required=True)
    
    groupSource = parser.add_argument_group('Data source')
    groupSource.add_argument('--bag', dest='bagfile', help='The bag file with the data')
    groupSource.add_argument('--topics', nargs='+', dest='topics', help='The list of image topics', required=True)
    groupSource.add_argument('--bag-from-to', metavar='bag_from_to', type=float, nargs=2, help='Use the bag data starting from up to this time [s]')
    groupSource.add_argument('--bag-freq', metavar='bag_freq', type=float, help='Frequency to extract features at [hz]')

    groupTarget = parser.add_argument_group('Calibration target configuration')
    groupTarget.add_argument('--target', dest='targetYaml', help='Calibration target configuration as yaml file', required=True)
    
    outputSettings = parser.add_argument_group('Output options')
    outputSettings.add_argument('--verbose', action='store_true', dest='verbose', help='Enable (really) verbose output (disables plots)')
    outputSettings.add_argument('--show-extraction', action='store_true', dest='showextraction', help='Show the calibration target extraction. (disables plots)')
    outputSettings.add_argument('--plot', action='store_true', dest='plot', help='Plot during calibration (this could be slow).')
    outputSettings.add_argument('--dont-show-report', action='store_true', dest='dontShowReport', help='Do not show the report on screen after calibration.')
    outputSettings.add_argument('--export-poses', action='store_true', dest='exportPoses', help='Export the optimized poses into a CSV (time_ns, position, quaterion)')

    #print help if no argument is specified
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(2)
        
    #Parser the argument list
    try:
        parsed = parser.parse_args()
    except:
        sys.exit(2)
    
    #some checks
    if len(parsed.topics) != len(parsed.models):
        sm.logError("Please specify exactly one camera model (--models) for each topic (--topics).")
        sys.exit(2)
    
    #there is a with the gtk plot widget, so we cant plot if we have opencv windows open...
    #--> disable the plots in these special situations
    if parsed.showextraction or parsed.verbose:
        parsed.dontShowReport = True
    
    return parsed


class AsyncCalibrator():
    # camera config, target config, dataset
    def __init__(self, cam0Config, cam1Config, targetConfig, dataset0, dataset1, reprojectionSigma=1.0, showCorners=True, showReproj=True, showOneStep=False):
        self.targetConfig = targetConfig

        self.cornerUncertainty = reprojectionSigma

        #set the extrinsic prior to default
        self.T_cam1_cam0 = sm.Transformation()

        #initialize timeshift prior to zero
        self.timeshiftCam0ToCam1Prior = 0.0

        ## camera 0
        self.cam0Config = cam0Config
        self.dataset0 = dataset0
        self.camera0 = kc.AslamCamera.fromParameters( self.cam0Config )   #load camera model from yaml
        self.setupCalibrationTarget( targetConfig, self.camera0, showExtraction=showCorners, showReproj=showReproj, imageStepping=showOneStep )
        multithreading = not (showCorners or showReproj or showOneStep)  #parallel processing for corner extraction if not visualizing
        self.cam0_detector = self.detector
        self.target0Observations = kc.extractCornersFromDataset(self.dataset0, self.cam0_detector, multithreading=multithreading) #T_t_c0 pose (target pose relative to cam0 frame thru PnP)

        ## camera 1
        self.cam1Config = cam1Config
        self.dataset1 = dataset1
        self.camera1 = kc.AslamCamera.fromParameters( self.cam1Config )
        self.setupCalibrationTarget( targetConfig, self.camera1, showExtraction=showCorners, showReproj=showReproj, imageStepping=showOneStep )
        multithreading = not (showCorners or showReproj or showOneStep)
        self.cam1_detector = self.detector
        self.target1Observations = kc.extractCornersFromDataset(self.dataset1, self.cam1_detector, multithreading=multithreading)  #T_t_c1 pose (target pose relative to cam1 frame thru PnP)

        print("Cam0: %d/%d images had valid detections (detector + PnP)" % (len(self.target0Observations), self.dataset0.numImages()))
        print("Cam1: %d/%d images had valid detections (detector + PnP)" % (len(self.target1Observations), self.dataset1.numImages()))
        
        # problem
        # self.problem = aopt.OptimizationProblem()
        self.problem = inc.CalibrationOptimizationProblem()

    # from IccSensors.py --> class IccCamera 
    def setupCalibrationTarget(self, targetConfig, camera, showExtraction=False, showReproj=False, imageStepping=False):
        
        #load the calibration target configuration
        targetParams = targetConfig.getTargetParams()
        targetType = targetConfig.getTargetType()
    
        if targetType == 'checkerboard':
            options = acv.CheckerboardOptions() 
            options.filterQuads = True
            options.normalizeImage = True
            options.useAdaptiveThreshold = True        
            options.performFastCheck = False
            options.windowWidth = 5
            options.showExtractionVideo = showExtraction
            grid = acv.GridCalibrationTargetCheckerboard(targetParams['targetRows'], 
                                                            targetParams['targetCols'], 
                                                            targetParams['rowSpacingMeters'], 
                                                            targetParams['colSpacingMeters'],
                                                            options)
        elif targetType == 'circlegrid':
            options = acv.CirclegridOptions()
            options.showExtractionVideo = showExtraction
            options.useAsymmetricCirclegrid = targetParams['asymmetricGrid']
            grid = acv.GridCalibrationTargetCirclegrid(targetParams['targetRows'],
                                                          targetParams['targetCols'], 
                                                          targetParams['spacingMeters'], 
                                                          options)
        elif targetType == 'aprilgrid':
            options = acv_april.AprilgridOptions() 
            options.showExtractionVideo = showExtraction
            options.minTagsForValidObs = int( np.max( [targetParams['tagRows'], targetParams['tagCols']] ) + 1 )
            
            grid = acv_april.GridCalibrationTargetAprilgrid(targetParams['tagRows'],
                                                            targetParams['tagCols'], 
                                                            targetParams['tagSize'], 
                                                            targetParams['tagSpacing'], 
                                                            options)
        else:
            raise RuntimeError( "Unknown calibration target." )
                          
        options = acv.GridDetectorOptions() 
        options.imageStepping = imageStepping
        options.plotCornerReprojection = showReproj
        options.filterCornerOutliers = True
        options.filterCornerSigmaThreshold = 2.0
        options.filterCornerMinReprojError = 0.2
        self.detector = acv.GridDetector(camera.geometry, grid, options)     

    #initialize a pose spline using camera poses (pose spline = T_wb)
    def initPoseSpline(self, targetObservations, splineOrder=6, poseKnotsPerSecond=200, timeOffsetPadding=0.02, label='cam'):
        '''
        poseKnotsPerSecond: number of knots per second for the pose spline (200 => 20ms)
        Eg:
            T_t_c: target pose relative to cam frame (from PnP)
            (T_t_c).T(): cam0 poses w.r.t target frame, for building spline (discrete poses)
        '''
        pose = bsplines.BSplinePose(splineOrder, sm.RotationVector() )
                
        # Get the grid detected times
        cam_times = np.array([obs.time().toSec() for obs in targetObservations])       #cam detection discrete timestamps in seconds      
        cam_curves = np.matrix([ pose.transformationToCurveValue(obs.T_t_c().T()) for obs in targetObservations]).T   #4x4 transformation matrix --> 6xN axis angle representation [tx, ty, tz, rx, ry, rz], each column = discrete pose 
        
        if np.isnan(cam_curves).any():
            raise RuntimeError("Nans in cam_pose values for initPoseSpline")
            sys.exit(0)
        
        ''' 
        Add 2 seconds on either end to allow the spline to slide during optimization

        time:
            before: t[0], t[1], t[2], ..., t[N]
            after:  t[0]-0.04, t[0], t[1], t[2], ..., t[N], t[N]+0.04

        curve values:
            before: pose[0], pose[1], pose[2], ..., pose[N]
            after:  pose[0], pose[0], pose[1], pose[2], ..., pose[N], pose[N]
       
        '''
        cam_times = np.hstack((cam_times[0] - (timeOffsetPadding * 2.0), cam_times, cam_times[-1] + (timeOffsetPadding * 2.0)))   #timeoffsetpadding(0.02) * 2 seconds = 0.04 = 40 ms on either end
        cam_curves = np.hstack((cam_curves[:,0], cam_curves, cam_curves[:,-1]))
        
        # Make sure the rotation vector doesn't flip
        for i in range(1,cam_curves.shape[1]):
            previousRotationVector = cam_curves[3:6,i-1]
            r = cam_curves[3:6,i]
            angle = np.linalg.norm(r)
            axis = r/angle
            best_r = r
            best_dist = np.linalg.norm( best_r - previousRotationVector)
            
            for s in range(-3,4):
                aa = axis * (angle + math.pi * 2.0 * s)   #angle wrapping --> axis*(angle + 2pi*s)
                dist = np.linalg.norm( aa - previousRotationVector )
                dist_neg = np.linalg.norm( -aa - previousRotationVector )   #axis flipped case 
                if dist < best_dist:
                    best_r = aa
                    best_dist = dist
                if dist_neg < best_dist:
                    best_r = -aa
                    best_dist = dist_neg
            cam_curves[3:6,i] = best_r;
        
        # discrete cam0 poses from PnP 
        np.savetxt('/data/poses_discrete_%s.csv' % label, 
            np.hstack((cam_times.reshape(-1,1), cam_curves.T)), 
            delimiter=',', 
            header='t,tx,ty,tz,rx,ry,rz')

        # Fitting the spline    
        seconds = cam_times[-1] - cam_times[0]
        knots = int(round(seconds * poseKnotsPerSecond))
        
        print("")
        print("Initializing a pose spline with %d knots (%f knots per second over %f seconds)" % ( knots, poseKnotsPerSecond, seconds))
        pose.initPoseSplineSparse(cam_times, cam_curves, knots, 1e-4)
        
        # Dense sampling to visualize smoothness
        dense_times = np.linspace(cam_times[0], cam_times[-1], 5000)
        dense_curves = np.array([pose.eval(t) for t in dense_times]).T
        np.savetxt('/data/poses_spline_dense_%s.csv' % label,
                np.hstack((dense_times.reshape(-1,1), dense_curves.T)),
                delimiter=',',
                header='t,tx,ty,tz,rx,ry,rz')
        
        return pose   
    
    def findTimeshiftPrior(self, verbose=False):
        print("Estimating time shift camera 1 to camera 0:")
        
        #fit a spline to the camera observations
        poseSplineCam0 = self.initPoseSpline( self.target0Observations, timeOffsetPadding=0.0, label='cam0_prior')
        poseSplineCam1 = self.initPoseSpline( self.target1Observations, timeOffsetPadding=0.0, label='cam1_prior' )
        
        omega_cam0_norm = []
        omega_cam1_norm = []

        # Uniform time grid over overlapping range
        t_start = max(poseSplineCam0.t_min(), poseSplineCam1.t_min())
        t_end = min(poseSplineCam0.t_max(), poseSplineCam1.t_max())
        dT = 0.01  # 10ms grid spacing
        uniform_times = np.arange(t_start, t_end, dT)

        for tk in uniform_times:
            omega0 = aopt.EuclideanExpression(np.matrix(poseSplineCam0.angularVelocityBodyFrame(tk)).transpose())
            omega1 = aopt.EuclideanExpression(np.matrix(poseSplineCam1.angularVelocityBodyFrame(tk)).transpose())
            omega_cam0_norm.append(np.linalg.norm(omega0.toEuclidean()))
            omega_cam1_norm.append(np.linalg.norm(omega1.toEuclidean()))

        omega_cam0_norm = np.array(omega_cam0_norm)
        omega_cam1_norm = np.array(omega_cam1_norm)

        #verify
        if len(omega_cam1_norm) == 0 or len(omega_cam0_norm) == 0:
            sm.logFatal("The time ranges of the camera 0 and camera 1 do not overlap. "\
                        "Please make sure that your sensors are synchronized correctly.")
            sm.logFatal("Cam0 spline range: {0} to {1}".format(poseSplineCam0.t_min(), poseSplineCam0.t_max()))
            sm.logFatal("Cam1 spline range: {0} to {1}".format(poseSplineCam1.t_min(), poseSplineCam1.t_max()))
            sm.logFatal("Cam0 first observation: {0}".format(self.target0Observations[0].time().toSec()))
            sm.logFatal("Cam0 last observation: {0}".format(self.target0Observations[-1].time().toSec()))
            sm.logFatal("Cam1 first observation: {0}".format(self.target1Observations[0].time().toSec()))
            sm.logFatal("Cam1 last observation: {0}".format(self.target1Observations[-1].time().toSec()))
            sys.exit(-1)

        # Cross-correlate
        corr = np.correlate(omega_cam1_norm, omega_cam0_norm, "full")
        discrete_shift = corr.argmax() - (len(omega_cam0_norm) - 1)
        shift = -discrete_shift * dT
        
        #Create plots
        if verbose:
            pl.plot(uniform_times, omega_cam0_norm, label="measured_raw")
            pl.plot(uniform_times, omega_cam1_norm, label="predicted")
            pl.plot(uniform_times-shift, omega_cam0_norm, label="measured_corrected")
            pl.legend()
            pl.xlabel("Lag (samples)")
            pl.ylabel("Correlation (sum of products of angular velocity rad^2/s^2)")
            pl.title("Time shift prior cam0-cam1 estimation")
            pl.figure()
            pl.plot(corr)
            pl.title("Cross-correlation ||omega_cam1||, ||omega_cam0||")
            pl.show()
            sm.logDebug("discrete time shift: {0}".format(discrete_shift))
            sm.logDebug("cont. time shift: {0}".format(shift))
            sm.logDebug("dT: {0}".format(dT))
        
        #store the timeshift (t_cam0 = t_cam1 + timeshiftCam0ToCam1Prior)
        self.timeshiftCam0ToCam1Prior = shift
        
        print("  Time shift camera 0 to camera 1 (t_cam1 = t_cam0 + shift):")
        print(self.timeshiftCam0ToCam1Prior)

    def addDesignVariables(self, dvc, noTimeCalibration=True, setActive=True, baselinedv_group_id=HELPER_GROUP_ID, calibration_group_id=CALIBRATION_GROUP_ID):
        '''
        design variables:
            T_cam1_cam0 extrinsic transformation
            T_cam0_cam0 identity transformation for cam0 (for building expressions) --> not needed in general but problem demands all design variables to be registered
            time shift DV
            cam0 spline control points (pose trajectory)
        '''

        #extrinsic transformation (divided into rotation and translation design variables to avoid singularities in the rotation representation)
        self.T_cam1_cam0_Dv = aopt.TransformationDv(self.T_cam1_cam0, rotationActive=True, translationActive=True)
        for i in range(0, self.T_cam1_cam0_Dv.numDesignVariables()):
            self.problem.addDesignVariable( self.T_cam1_cam0_Dv.getDesignVariable(i), baselinedv_group_id)

        #identity transformation (expression format for cam0)
        self.T_cam0_cam0_Dv = aopt.TransformationDv(sm.Transformation(), rotationActive=False, translationActive=False)
        for i in range(0, self.T_cam0_cam0_Dv.numDesignVariables()):
            self.problem.addDesignVariable( self.T_cam0_cam0_Dv.getDesignVariable(i), baselinedv_group_id)
        
        #time delay design variable
        self.cam0TimeTocam1TimeDV = aopt.Scalar(0.0)
        self.cam0TimeTocam1TimeDV.setActive(not noTimeCalibration)
        self.problem.addDesignVariable(self.cam0TimeTocam1TimeDV, calibration_group_id)

        #spline design variable
        for i in range(0, dvc.numDesignVariables()):
            dv = dvc.designVariable(i)
            dv.setActive(setActive)
            self.problem.addDesignVariable(dv, baselinedv_group_id)

        # dvc = asp.BSplinePoseDesignVariable(poseSpline)

    def addCameraErrorTerms(self, dataset, camera, targetobservations, T_camN_cam0, poseSplineDv=None, blakeZissermanDf=0.0, timeOffsetPadding=0.0, applyFrameTimeShift=False):
        print("")
        print("Adding camera error terms ({0})".format(dataset.topic))

        #progress bar
        iProgress = sm.Progress2(len(targetobservations))
        iProgress.sample()

        allReprojectionErrors = list()   #store reprojection errors 
        error_t = camera.reprojectionErrorType

        for obs in targetobservations:
            #Build a transformation expression for this time
            if applyFrameTimeShift:
                #for cam1 
                frameTime = self.cam0TimeTocam1TimeDV.toExpression() + obs.time().toSec() + self.timeshiftCam0ToCam1Prior  #expression
                frameTimeScalar = frameTime.toScalar()  #plain float 
            else:
                frameTime = aopt.ScalarExpression(obs.time().toSec()) #cam 0 being the reference, so no time shift (already float)
                frameTimeScalar = frameTime.toScalar()

            #check initial frame time is within the spline 
            if frameTimeScalar <= poseSplineDv.spline().t_min() or frameTimeScalar >= poseSplineDv.spline().t_max():
                continue
            
            #T_dest_src
            #T_w_cam0: from cam0 to world coords 
            T_w_cam0 = poseSplineDv.transformationAtTime(frameTime, timeOffsetPadding, timeOffsetPadding)  #get the transformation from the observations
            T_cam0_w = T_w_cam0.inverse()  #from world to cam0 coords

            #calibration target coords to camera N coords
            # T_w_cam0: from cam0 to world
            # T_camN_w: from world to cam N
            # T_camN_cam0: we get this from PnP
            T_camN_w = T_camN_cam0 * T_cam0_w
            
            #get the image and target points corresponding to the frame
            imageCornerPoints = np.array(obs.getCornersImageFrame()).T
            targetCornerPoints = np.array(obs.getCornersTargetFrame()).T

            #setup an aslam frame (handles the distortion)
            frame = camera.frameType()
            frame.setGeometry(camera.geometry)

            #corner uncertainty
            R = np.eye(2) * self.cornerUncertainty * self.cornerUncertainty
            invR = np.linalg.inv(R)

            for pidx in range(0, imageCornerPoints.shape[1]):
                #add all image points
                k = camera.keypointType()
                k.setMeasurement(imageCornerPoints[:, pidx])
                k.setInverseMeasurementCovariance(invR)
                frame.addKeypoint(k)

            reprojectionErrors = list()
            for pidx in range(0, imageCornerPoints.shape[1]):
                #add all target points
                targetPoint = np.insert(targetCornerPoints.transpose()[pidx], 3, 1)
                p = T_camN_w * aopt.HomogeneousExpression(targetPoint)  #from target frame to camN frame (cam0 if camN_cam0 is identity)

                #build and append the error term
                rerr = error_t(frame, pidx, p)

                #add blake-zisserman m-estimator
                if blakeZissermanDf>0.0:
                    mest = aopt.BlakeZissermanMEstimator( blakeZissermanDf )
                    rerr.setMEstimatorPolicy(mest)
                
                self.problem.addErrorTerm(rerr)  
                reprojectionErrors.append(rerr)
            
            allReprojectionErrors.append(reprojectionErrors)
                        
            #update progress bar
            iProgress.sample()
            
        print("\r  Added {0} camera error terms                      ".format( len(targetobservations) ))           

        return allReprojectionErrors

    def buildAndSolveProblem(self, 
                             splineOrder=6, 
                             poseKnotsPerSecond=70,
                             maxIterations=20,
                             timeOffsetPadding=0.02,
                             blakeZisserCam=-1,
                             verbose=False
                             ):
        
        print("\tSpline order: %d" % (splineOrder))
        print("\tPose knots per second: %d" % (poseKnotsPerSecond))
        print("\tMax iterations: %d" % (maxIterations))
        print("\tTime offset padding: %f" % (timeOffsetPadding))

        #######################
        ## timeshift prior
        #######################
        self.findTimeshiftPrior(verbose=verbose)

        #######################
        ## cam0 pose spline initialized 
        #######################
        poseSpline = self.initPoseSpline(self.target0Observations, splineOrder, poseKnotsPerSecond, timeOffsetPadding, label="cam0_main")
        
        # initialize design variables
        poseSplineDv = asp.BSplinePoseDesignVariable( poseSpline )
        self.addDesignVariables(poseSplineDv, noTimeCalibration=False, setActive=True)

        # problem = inc.CalibrationOptimizationProblem()

        #######################
        ## add error terms
        #######################

        #cam0 reprojection error terms (cam0 is the reference, so no time shift)
        self.cam0ReprojectionErrors = self.addCameraErrorTerms(self.dataset0, self.camera0, self.target0Observations, self.T_cam0_cam0_Dv.toExpression(), poseSplineDv, 
                                 blakeZissermanDf=blakeZisserCam, timeOffsetPadding=0.0, applyFrameTimeShift=False) 
        
        
        #cam1 reprojection error terms (apply time shift to cam1)
        self.cam1ReprojectionErrors = self.addCameraErrorTerms(self.dataset1, self.camera1, self.target1Observations, self.T_cam1_cam0_Dv.toExpression(), poseSplineDv, 
                                 blakeZissermanDf=blakeZisserCam, timeOffsetPadding=timeOffsetPadding, applyFrameTimeShift=True)
        
        ######################
        ## solve the problem
        ######################
        options = aopt.Optimizer2Options()
        options.verbose = verbose
        options.doLevenbergMarquardt = True
        options.levenbergMarquardtLambdaInit = 10.0
        options.nThreads = max(1, multiprocessing.cpu_count()-1)
        options.convergenceDeltaX = 1e-5
        options.convergenceDeltaJ = 1e-2
        options.maxIterations = maxIterations
        options.trustRegionPolicy = aopt.LevenbergMarquardtTrustRegionPolicy(options.levenbergMarquardtLambdaInit)
        options.linearSolver = aopt.BlockCholeskyLinearSystemSolver()

        self.optimizer = aopt.Optimizer2(options)
        self.optimizer.setProblem(self.problem)

        try:
            self.optimizer.optimize()
        except Exception as e:
            sm.logError(str(e))
            raise RuntimeError("Optimization failed!")
        
        ########################
        ## return results
        ########################
        return sm.Transformation(self.T_cam1_cam0_Dv.T()), self.cam0TimeTocam1TimeDV.toScalar() + self.timeshiftCam0ToCam1Prior
    

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_exit)
    
    bagfile = '/data/merged_payload4b/decompressed_ros1.bag'
    cam0YamlFile = '/data/cam0.yaml'
    cam1YamlFile = '/data/cam1.yaml'
    targetYamlFile = '/data/aprilgrid.yaml'

    # load configs
    cam0Config = kc.CameraParameters(cam0YamlFile)
    cam1Config = kc.CameraParameters(cam1YamlFile)
    targetConfig = kc.CalibrationTargetParameters(targetYamlFile)

    # load datasets
    dataset0 = initBagDataset(bagfile, '/cam_sync/cam0/image_raw', None, None)
    dataset1 = initBagDataset(bagfile, '/cam_sync/cam1/image_raw', None, None)

    # initialize calibrator
    calibrator = AsyncCalibrator(cam0Config, cam1Config, targetConfig, dataset0, dataset1,
                                 reprojectionSigma=1.0, 
                                 showCorners=False, 
                                 showReproj=False, 
                                 showOneStep=False)

    # build and solve
    T_cam1_cam0, timeShiftCam0ToCam1 = calibrator.buildAndSolveProblem(
                                            splineOrder=6, 
                                            poseKnotsPerSecond=70, 
                                            maxIterations=20, 
                                            timeOffsetPadding=0.02, 
                                            blakeZisserCam=-1, 
                                            verbose=True)

    print("")
    print("Calibration results:")
    print("T_cam1_cam0:")
    print(T_cam1_cam0.T())
    print("Time shift cam0 to cam1 (t_cam1 = t_cam0 + shift):")
    print(timeShiftCam0ToCam1)


        

                                 











