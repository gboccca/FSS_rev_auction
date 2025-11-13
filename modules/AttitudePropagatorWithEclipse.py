import math
import numpy as np

"""
This script provides functionality for propagating the attitude of a satellite using quaternions and calculating 
the satellite's pointing direction. It consists of three main functions:

1. `propagate_attitude(self, time_step)`:
    - This function propagates the attitude quaternion using quaternion kinematics and Euler integration.
    - The attitude quaternion is updated based on the satellite's angular velocity and then normalized 
      to maintain unit length, ensuring the quaternion remains valid.
    - The satellite's attitude quaternion is stored and updated in the object's `self.attitude` attribute.

2. `calculate_pointing_direction(self)`:
    - This function calculates the satellite's pointing direction based on its current attitude quaternion.
    - It converts the quaternion to a rotation matrix, and the pointing direction is extracted as the third 
      column of the rotation matrix, which corresponds to the satellite's orientation.

3. `quaternion_to_rotation_matrix(quaternion)`:
    - This static method converts a given quaternion to a 3x3 rotation matrix.
    - The rotation matrix is used to describe the satellite's orientation in space, with the third column 
      indicating its pointing direction.

These functions use quaternion kinematics for attitude propagation, a method commonly used in spacecraft dynamics 
for avoiding singularities associated with Euler angles.
"""

def propagate_attitude(satellite, time_step):
    # Propagate the attitude quaternion using quaternion kinematics
    quaternion = satellite.attitude['quaternion']
    angular_velocity = satellite.attitude['angular_velocity']

    # Update quaternion using quaternion kinematics (Euler integration)
    q_dot = 0.5 * np.array([
        -quaternion[1] * angular_velocity[0] - quaternion[2] * angular_velocity[1] - quaternion[3] * angular_velocity[2],
            quaternion[0] * angular_velocity[0] + quaternion[2] * angular_velocity[2] - quaternion[3] * angular_velocity[1],
            quaternion[0] * angular_velocity[1] - quaternion[1] * angular_velocity[2] + quaternion[3] * angular_velocity[0],
            quaternion[0] * angular_velocity[2] + quaternion[1] * angular_velocity[1] - quaternion[2] * angular_velocity[0],
    ])
    quaternion += q_dot * time_step

    # Normalize quaternion to maintain unit length
    quaternion /= np.linalg.norm(quaternion)

    # Update attitude parameters
    satellite.attitude['quaternion'] = quaternion


def calculate_pointing_direction(satellite):
    # Calculate pointing direction from the attitude quaternion
    quaternion = satellite.attitude['quaternion']

    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)

    # Extract pointing direction (third column of rotation matrix)
    pointing_direction = rotation_matrix[:, 2]

    return pointing_direction

def is_in_eclipse(satellite, sun_position):
        """
        Determine if the satellite is in the Earth's shadow (eclipse).
        
        Parameters:
            sun_position (tuple): The position of the Sun in ECI coordinates.
        
        Returns:
            bool: True if the satellite is in eclipse, False otherwise.
        """
        # earth_radius = 6371  # Radius of the Earth in kilometers
        # satellite_position = np.array([satellite.orbit['x'], satellite.orbit['y'], satellite.orbit['z']])/1000
        # #print("satellite_position",satellite_position)
        # sun_position = np.array(sun_position)
        # #print("sun norm is: ", np.linalg.norm(sun_position))

        # # Vector from satellite to Earth's center
        # sat_to_earth = -satellite_position
        
        # # Vector from satellite to Sun
        # sat_to_sun = sun_position - satellite_position
        
        # # Check if satellite is in the shadow cone
        # earth_shadow_cone_angle = np.arcsin(earth_radius / np.linalg.norm(sun_position))
        
        # # Angle between sat_to_earth and sat_to_sun
        # angle = np.arccos(np.dot(sat_to_earth, sat_to_sun) / (np.linalg.norm(sat_to_earth) * np.linalg.norm(sat_to_sun)))
        # # print("Satellite position:", satellite_position)
        # # print("Sun position:", sun_position)
        # # print("Satellite to Earth vector:", sat_to_earth)
        # # print("Satellite to Sun vector:", sat_to_sun)
        # # print("Earth shadow cone angle:", earth_shadow_cone_angle)
        # # print("Angle between Earth and Sun vectors:", angle)
        #return angle < earth_shadow_cone_angle



        # earth_radius = 6371  # Earth's radius in kilometers
        # satellite_position = np.array([satellite.orbit['x'], satellite.orbit['y'], satellite.orbit['z']]) / 1000  # km
        # sun_position = np.array(sun_position) / 1000  # km

        # # Vector from Earth's center to the Sun
        # earth_to_sun_vector = sun_position

        # # Vector from Earth's center to the satellite
        # earth_to_satellite_vector = satellite_position

        # # Calculate the distance from Earth to Sun and Earth to Satellite
        # earth_to_sun_distance = np.linalg.norm(earth_to_sun_vector)
        # earth_to_satellite_distance = np.linalg.norm(earth_to_satellite_vector)

        # # Calculate Earth's angular radius as seen from the satellite
        # earth_angular_radius = np.arcsin(earth_radius / earth_to_satellite_distance)

        # # Calculate the angle between Earth-Sun and Earth-Satellite vectors
        # angle_between_vectors = np.arccos(
        #     np.dot(earth_to_sun_vector, earth_to_satellite_vector) / (earth_to_sun_distance * earth_to_satellite_distance)
        # )
        # print("Satellite position:", satellite_position)
        # print("Sun position:", sun_position)
        # print("Earth shadow cone angle:", earth_angular_radius)
        # print("Angle between Earth and Sun vectors:", angle_between_vectors)
        # return angle_between_vectors < earth_angular_radius
        earth_radius = 6371  # Radius of the Earth in kilometers
        sun_radius = 696340  # Radius of the Sun in kilometers

        # Calculate the length of the Earth's shadow cone
        shadow_cone_length = (earth_radius / (sun_radius - earth_radius)) * np.linalg.norm(sun_position)

        # Get satellite and sun positions in kilometers
        satellite_position = np.array([satellite.orbit['x'], satellite.orbit['y'], satellite.orbit['z']]) / 1000  # km
        sun_position = np.array(sun_position) / 1000  # km

        # Vector from Earth's center to Sun and to Satellite
        earth_to_sun_vector = sun_position
        earth_to_satellite_vector = satellite_position

        # Calculate distances
        earth_to_sun_distance = np.linalg.norm(earth_to_sun_vector)
        earth_to_satellite_distance = np.linalg.norm(earth_to_satellite_vector)

        # Check if the satellite is beyond the shadow cone length
        if earth_to_satellite_distance > shadow_cone_length:
            return False  # Satellite is too far to be in the shadow cone

        # Calculate Earth's angular radius as seen from the satellite
        earth_angular_radius = np.arcsin(earth_radius / earth_to_satellite_distance)

        # Calculate the angle between Earth-Sun and Earth-Satellite vectors
        angle_between_vectors = np.arccos(
            np.dot(earth_to_sun_vector, earth_to_satellite_vector) / (earth_to_sun_distance * earth_to_satellite_distance)
        )

        # Satellite is in eclipse if the angle is less than Earth's angular radius as seen from the satellite
        return angle_between_vectors < earth_angular_radius

@staticmethod    
def quaternion_to_rotation_matrix(quaternion):
    q = quaternion
    return np.array([
        [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[0]*q[2] + 2*q[1]*q[3]],
        [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
        [-2*q[0]*q[2] + 2*q[1]*q[3], 2*q[0]*q[1] + 2*q[2]*q[3], 1 - 2*q[1]**2 - 2*q[2]**2]
    ])

def sun_position(time):
    """
    Approximate the position of the Sun in ECI coordinates.
    
    Parameters:
        time (float): The time in seconds since the epoch.
    
    Returns:
        tuple: The position of the Sun in ECI coordinates.
    """
    # Assuming a simple circular orbit of the Earth around the Sun
    orbit_radius = 1.496e8  # Average distance from Earth to Sun in kilometers
    orbital_period = 365.25 * 24 * 3600  # One year in seconds

    # Calculate the position of the Sun
    theta = 2 * math.pi * (time % orbital_period) / orbital_period
    x_Sun= orbit_radius * math.cos(theta)
    y_Sun = orbit_radius * math.sin(theta)
    z_Sun = 0  # Simplification: assuming the orbit is in the xy-plane
    return x_Sun, y_Sun, z_Sun