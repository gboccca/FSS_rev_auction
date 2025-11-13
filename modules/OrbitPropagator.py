import math

def propagate_orbit(satellite, time, update_position=True):
    """Propagate the orbit of the satellite and return position and velocity."""
    # Read orbital parameters from the satellite's orbit dictionary
    a = satellite.orbit['semimajoraxis'] * 1000  # Semimajor axis in meters
    e = satellite.orbit['eccentricity']
    i = math.radians(satellite.orbit['inclination'])
    omega = math.radians(satellite.orbit['arg_of_perigee'])
    Omega = math.radians(satellite.orbit['raan'])
    theta0 = math.radians(satellite.orbit['true_anomaly'])

    # Mean motion and gravitational parameter
    mu = 3.986004418e14  # Earth's gravitational parameter in m^3/s^2
    n = math.sqrt(mu / a ** 3)  # Mean motion in radians per second

    # Calculate Mean anomaly at the specified time
    M0 = theta0 - e * math.sin(theta0)
    M = M0 + n * time

    # Solve Kepler's equation for the eccentric anomaly E
    E = M
    for _ in range(10):  # Iterative solution to refine E
        E = M + e * math.sin(E)

    # True anomaly
    theta = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))

    # Radius
    r = a * (1 - e ** 2) / (1 + e * math.cos(theta))

    # Position in inertial frame
    x = r * (math.cos(Omega) * math.cos(omega + theta) - math.sin(Omega) * math.sin(omega + theta) * math.cos(i))
    y = r * (math.sin(Omega) * math.cos(omega + theta) + math.cos(Omega) * math.sin(omega + theta) * math.cos(i))
    z = r * math.sin(omega + theta) * math.sin(i)

    # Velocity in inertial frame
    v_x = -n * r / math.sqrt(1 - e ** 2) * (math.cos(Omega) * math.sin(omega + theta) + math.sin(Omega) * math.cos(omega + theta) * math.cos(i))
    v_y = n * r / math.sqrt(1 - e ** 2) * (math.sin(Omega) * math.sin(omega + theta) - math.cos(Omega) * math.cos(omega + theta) * math.cos(i))
    v_z = n * r / math.sqrt(1 - e ** 2) * math.sin(i)

    # Update satellite's position if requested
    if update_position:
        satellite.orbit['x'], satellite.orbit['y'], satellite.orbit['z'] = x, y, z
        satellite.orbit['vx'], satellite.orbit['vy'], satellite.orbit['vz'] = v_x, v_y, v_z

    # Return position and velocity as tuples
    return (x, y, z), (v_x, v_y, v_z)

def distance_between(sat1, sat2, time_step):
    """Calculate distance between two satellites by propagating both to the same time."""
    # Propagate each satellite to get their positions at the same time
    pos1, _ = propagate_orbit(sat1, time_step)
    pos2, _ = propagate_orbit(sat2, time_step)

    # Calculate Euclidean distance between positions
    dx, dy, dz = pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return distance