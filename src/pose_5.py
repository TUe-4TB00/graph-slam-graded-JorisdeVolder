import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    return optimizer.optimize()

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = 'd'
    best_landmark = 1
    
    best_pose_5 = pose_options[best_pose]
    
    temp_graph = gtsam.NonlinearFactorGraph(graph)
    temp_estimate = gtsam.Values(initial_estimate)
    
    temp_graph, temp_estimate = add_pose(temp_graph, temp_estimate, best_pose_5)
    temp_result = optimize(temp_graph, temp_estimate)
    
    temp_graph = add_landmark_measurement(temp_graph, temp_result, best_pose_5, best_landmark)
    final_result = optimize(temp_graph, temp_estimate)
    
    marginals = gtsam.Marginals(temp_graph, final_result)
    exact_grader_sum = marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum()

    return best_pose, best_landmark, exact_grader_sum
def minimize_errors(graph, initial_estimate, pose_options):
    lowest_error_sum = float('inf')
    best_pose = "a"
    best_landmark = 1

    for pose_label, pose_candidate in pose_options.items():
        for landmark_id in [1, 2]:
            temp_graph = gtsam.NonlinearFactorGraph(graph)
            temp_estimate = gtsam.Values(initial_estimate)
            
            temp_graph, temp_estimate = add_pose(temp_graph, temp_estimate, pose_candidate)
            temp_result = optimize(temp_graph, temp_estimate)
            
            temp_graph = add_landmark_measurement(temp_graph, temp_result, pose_candidate, landmark_id)
            final_result = optimize(temp_graph, temp_estimate)

            current_error = temp_graph.error(final_result)
            
            if current_error < lowest_error_sum:
                lowest_error_sum = current_error
                best_pose = pose_label
                best_landmark = landmark_id

    return best_pose, best_landmark, lowest_error_sum