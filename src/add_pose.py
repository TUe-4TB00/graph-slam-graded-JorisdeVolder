import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):

    dx = math.sqrt(2.0)
    dy = math.sqrt(2.0)
    dtheta = math.pi / 2.0 
    
    odometry_measurement = gtsam.Pose2(dx, dy, dtheta)
    
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry_measurement, ODOMETRY_NOISE))
    
    x_4_global = 4.0 + dx
    y_4_global = dy
    theta_4_global = dtheta
    
    pose_4_guess = gtsam.Pose2(x_4_global, y_4_global, theta_4_global)
    
    initial_estimate.insert(X(4), pose_4_guess)
    
    return graph, initial_estimate