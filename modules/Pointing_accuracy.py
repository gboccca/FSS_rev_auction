import numpy as np
from modules.AttitudePropagatorWithEclipse import calculate_pointing_direction

def evaluate_pointing_accuracy(observer_satellite, target_satellite):
    # Calculate the pointing direction of the observer satellite
    pointing_direction = calculate_pointing_direction(satellite=observer_satellite)

    # Get the position vector of the target satellite
    target_position = np.array([
        target_satellite.orbit['x'],
        target_satellite.orbit['y'],
        target_satellite.orbit['z']
    ])

    # Calculate the vector pointing from observer to target
    observer_to_target_vector = target_position - np.array([
        observer_satellite.orbit['x'],
        observer_satellite.orbit['y'],
        observer_satellite.orbit['z']
    ])

    # Normalize the pointing direction vector and observer-to-target vector
    pointing_direction_norm = pointing_direction / np.linalg.norm(pointing_direction)
    observer_to_target_norm = observer_to_target_vector / np.linalg.norm(observer_to_target_vector)
    # if observer_to_target_norm == 0:
    #     print("Warning: Observer and target satellites are at the same position.")
    #     return np.nan  # Return NaN or some default error value

    # Calculate the cosine of the angle between the pointing direction and observer-to-target vector
    cos_angle = np.dot(pointing_direction_norm, observer_to_target_norm)

    # Calculate the angular distance (in radians) using the arccosine of the cosine of the angle
    angular_distance = np.arccos(cos_angle)

    # Convert angular distance from radians to degrees
    angular_distance_deg = np.degrees(angular_distance)

    # Print the angular distance as the pointing accuracy for the observer satellite
    #print(f"Observer satellite {observer_satellite.name} sees target satellite {target_satellite.name} with a "
     #     f"Pointing accuracy (angular distance) to target: {angular_distance_deg:.2f} degrees")
    
    return angular_distance_deg