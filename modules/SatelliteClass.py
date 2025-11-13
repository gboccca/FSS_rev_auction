# SatelliteClass.py

import math
import random
import numpy as np

class Satellite:
    def __init__(self, mode,
                 num_satellites_per_plane=None, num_planes=None,
                 orbit=None, epsys=None, commsys=None, DataHand=None,
                 PropSys=None, Optic=None, category=None, attitude=None,
                 availability=None, instrumentation=None, tasks=None, conflicts=None,
                 name=None, flag=None, index=None, total_satellites=None,
                 # NEW: constellation overrides
                 semimajoraxis_km=None, inclination_deg=None, walker_cfg=None):

        """
        mode: str, either 'SSO' for Sun-synchronous orbit or 'Walker' for Walker constellation
        i: int, satellite index
        num_satellites_per_plane: int, required for Walker mode, number of satellites per plane
        num_planes: int, required for Walker mode, number of orbital planes
        """
        self.processing_time = 0  # Add the processing_time attribute

        if mode == 'SSO':
            # Initialize with Sun-synchronous orbit (SSO) parameters
            semimajoraxis_sso = 7070  # Corresponding to 700-800 km altitudes
            J2 = 1.08263e-3  # Second zonal harmonic of Earth's gravitational potential
            omega_E = 7.2921159e-5  # rad/s, Earth's rotation rate
            R_Earth = 6378.137  # km, Mean Earth radius
            RAAN_ChangeSSO = 0.986

            # Calculate inclination for SSO
            cos_i = RAAN_ChangeSSO / (-9.96 * (R_Earth / semimajoraxis_sso) ** (7 / 2))
            inclination = math.acos(cos_i) * 180 / math.pi  # Convert from radians to degrees

            if orbit is None:
                orbit = {
                    'semimajoraxis': semimajoraxis_sso,  # km
                    'inclination': inclination,  # degrees
                    'eccentricity': 0,
                    'raan': index * (360 / total_satellites) if index is not None and total_satellites is not None else 0,
                    'arg_of_perigee': 0,
                    'true_anomaly': index * (360 / total_satellites) if index is not None and total_satellites is not None else 0,
                }



        elif ((mode == 'WALKER') or (mode == 'Walker')) and \
                (num_satellites_per_plane is not None) and (num_planes is not None):

            # Defaults if nothing is provided from the preset/simulation
            semimajoraxis = 6921 if semimajoraxis_km is None else float(semimajoraxis_km)  # ~550 km alt
            inclination = 53 if inclination_deg is None else float(inclination_deg)

            # Walker options: currently only phasing F (integer)
            cfg = walker_cfg or {}
            F = int(cfg.get('F', 0))

            # Determine plane/slot indices
            plane_idx = (index // num_satellites_per_plane) if index is not None else 0
            slot_idx = (index % num_satellites_per_plane) if index is not None else 0

            # Spacing
            raan_increment = 360.0 / float(num_planes)
            ta_increment = 360.0 / float(num_satellites_per_plane)

            # Walker-Delta phasing: TA offset per plane
            # true_anomaly = slot*Δ + (F * plane_idx * Δ)   (mod 360)
            phasing = (F * plane_idx * ta_increment) % 360.0
            true_anom = (slot_idx * ta_increment + phasing) % 360.0
            raan = (plane_idx * raan_increment) % 360.0
            if orbit is None:
                orbit = {
                    'semimajoraxis': semimajoraxis,  # km
                    'inclination': inclination,  # deg
                    'eccentricity': 0,
                    'raan': raan,  # deg
                    'arg_of_perigee': 0,
                    'true_anomaly': true_anom,  # deg
                }

        else:
            raise ValueError("Invalid mode or missing parameters for Walker constellation")


        # Electric Power Subsystem
        if epsys is None: 
            epsys = {
                #'EnergyAvailable': 84 * 3600,  # [J]
                'EnergyAvailable': random.randint(40 * 3600, 84 * 3600),  # Random energy in the range 40-84 kJ
                'EnergyStorage': 84 * 3600,
                'SolarPanelSize': 0.4 * 0.3,
                'OperatingPower': 20,  # [W]
                'EclipsePower': 5.0,  # [W]
                # Add more subsystems as needed
            }

        # Communication Subsystem
        if commsys is None:
            commsys = {
                'band': 1  # Default to UHF
            }

        # Data Handling
        if DataHand is None:
            DataHand = {
                'DataStorage': 8 * 64e9,  # Maximum storage onboard (bytes)
                'StorageAvailable': random.randint(int(1 * 64e9), int(8 * 64e9)),   # Available storage onboard (bytes)	
                #'StorageAvailable': 8 * 64e9,  # Available storage onboard (bytes)	

            }
        
        # Propulsion System
        if PropSys is None:
            PropSys = {
                'PropellantMass': 1,  # [kg]
                'PropulsionType': 0,   # 0 for chemical, 1 for electrical
                'SpecificImpulse': 250, # [s]
                'Thrust': 1,            # [N]
            }

        # Optical Payload 
        if Optic is None:
            Optic = {
                'ApertureDiameter': 0.09,  # [m]
                'Wavelength': 700e-9,      # [m]
            }
        
        # Category
        if category is None:
            category = {
                'Target': 0,        # Index for target identification
                'Observation': 1,   # Index for observation satellite
            }

        # Attitude Parameters
        if attitude is None:
            attitude = {
                'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),  # Initial quaternion [w, x, y, z]
                #'quaternion': np.array([0.0, 0.0, 0.0, 1.0]),  # Initial quaternion [w, x, y, z]

                'angular_velocity': np.array([0.0, 0.0, 0.0]),  # [rad/s]
            } 


             #Define availability of the satellite
        if availability is None:
            availability= {
            'availability': 1,   # use 1 for available and 0 for not available
            } 
        
        if instrumentation is None:
            instrumentation= {
            'cameraVIS': 1,
            'cameraIR': 1,
            'radar': 1,
            'GNSS': 1,
            } 

        # Tasks and Reward
        # 
        if tasks is None:
            tasks = {}

        if conflicts is None:
            conflicts = {}
        # give a unique identifier to each task, as a protocol, for universal identification (getting ridd of local task id)

        self.flags = {
            'dummy' : 1,        # dummy flag, all sats have 1
            'central': 0,       # 1 for central node, 0 for non-central   
            'tasks': 0,         # 1 if sat has received tasks, 0 if not
            'RF': 0,            # 1 if sat has calculated a local RF != 0, 0 if not
            'comm' : 0,         # 1 if sat has communicated last timestep, 0 if not
            }
        
        self.orbit = orbit
        self.epsys = epsys
        self.commsys = commsys
        self.DataHand = DataHand
        self.PropSys = PropSys
        self.Optic = Optic
        self.category = category
        self.attitude = attitude
        self.tasks = tasks
        self.name = name
        self.availability = availability	
        self.instrumentation = instrumentation
        self.flag = flag

        C_fixed_mu, C_fixed_sigma = 25.0, 5.0  # € per task
        C_var_km2_mu, C_var_km2_sigma = 2.0, 0.5  # €/km^2

        C_fixed = max(0.0, float(np.random.normal(C_fixed_mu, C_fixed_sigma)))
        C_var_km2 = max(1e-6, float(np.random.normal(C_var_km2_mu, C_var_km2_sigma)))

        # Store both €/km^2 and €/m^2 so either can be used downstream
        self.cost = {
            "C_fixed": C_fixed,
            "C_var_km2": C_var_km2,
            "C_var_m2": C_var_km2 * 1e-6,
        }

    def check_task_conflict(self):
        for task_id, task in self.tasks.items():
            if task.get('thinks_winner'):
                for other_task_id, other_task in self.tasks.items():
                    if other_task_id != task_id and other_task.get('thinks_winner'):
                        # Check if the tasks have to be executed around the same time:
                        if abs(task['global']['time'] - other_task['global']['time']) < 10:  # Assuming a conflict if tasks are within 10 seconds of each other
                            conflict_key = f"{task_id}_{other_task_id}"
                            if conflict_key not in self.conflicts:
                                self.conflicts[conflict_key] = {
                                    'task1': task_id,
                                    'task2': other_task_id,
                                    'time': max(task['global']['time'], other_task['global']['time']),
                                    'RF': max(task['global']['RF'], other_task['global']['RF']),
                                }
                                print(f"Conflict detected between tasks {task_id} and {other_task_id} at time {self.conflicts[conflict_key]['time']} with RF {self.conflicts[conflict_key]['RF']}")

                        

class TargetSatellite(Satellite):
    def __init__(self, orbit=None, *args, **kwargs):
        # Set default orbital parameters if none are provided
        # if orbit is None:
        #     orbit = {
        #         'semimajoraxis': 6784,  # default to 6784 km for ISS, change as needed
        #         'inclination': 51.64,    # 51.64 typical ISS inclination
        #         'eccentricity': 0.0009,  # 0.0009 for ISS orbit
        #         'raan': 20,              # right ascension of ascending node
        #         'arg_of_perigee': 138,    # argument of perigee
        #         'true_anomaly': 20,      # starting true anomaly
        #     }
        if orbit is None:
            orbit = {
                'semimajoraxis': 7171,  # default to 7171 km for Envisat, change as needed
                'inclination': 98.55,    # 98.55 typical Envisat inclination
                'eccentricity': 0.0009,  # 0.0009 for ISS orbit
                'raan': 267,              # right ascension of ascending node
                'arg_of_perigee': 84,    # argument of perigee
                'true_anomaly': 20,      # starting true anomaly
            }
        super().__init__(orbit=orbit, *args, **kwargs)
        self.category = "target"

class ObserverSatellite(Satellite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = "observer"
        # Any other specific attributes/methods for the observer satellite
