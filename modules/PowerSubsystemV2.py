import numpy as np
from modules.AttitudePropagatorWithEclipse import is_in_eclipse, sun_position, calculate_pointing_direction

class PowerSubsystem:

#In this class, you can simulate power generation from solar panels, power consumption,
#  energy storage in a battery, and checking the availability of power during each time step.
#  You can also recharge the battery with excess energy.

    def __init__(self, satellite):
            self.satellite = satellite
            self.energy = satellite.epsys['EnergyStorage']  # Start with the initial energy from satellite epsys
            self.battery_capacity = satellite.epsys['EnergyStorage']  # Use the same value or a different if specified
            self.solar_panel_efficiency = 0.22  # Typical efficiency of space-grade solar panels
            self.solar_flux = 1370  # Solar flux near Earth in W/m^2
            self.solar_panel_size = satellite.epsys['SolarPanelSize']  # Use the solar panel size from epsys
            self.solar_panel_packing = 0.9  # Packing factor of the solar panel

    def generate_power(self, satellite, time_step):
        if is_in_eclipse(satellite,sun_position(time_step)):  # Assuming `sun_position` function gives the sun's position at the given time
            generated_power = 0
        else:
            sun_vector = np.array(sun_position(time_step)) - np.array([satellite.orbit['x'], satellite.orbit['y'], satellite.orbit['z']])
            sun_vector /= np.linalg.norm(sun_vector)
            panel_normal = calculate_pointing_direction(satellite)
            #cos_angle = max(np.dot(panel_normal, sun_vector), 0)
            #rint(f"Cosine of angle between panel normal and sun vector: {cos_angle}")
            cos_angle = 1 # Assuming the solar panels are always facing the Sun
            generated_power = self.solar_panel_efficiency * self.solar_flux * self.solar_panel_size * self.solar_panel_packing * cos_angle * time_step
            #print(f"Generated power: {generated_power} Wh")
        # Add generated power to the battery, ensuring it does not exceed capacity
        self.energy = min(self.energy + generated_power, self.battery_capacity)

    def consume_power(self, power_consumption, time_step):
        # Calculate the power consumption over the given time step
        consumed_power = power_consumption * time_step
        #print(f"Consumed power: {consumed_power} Wh")
        self.energy -= consumed_power
        # Ensure the battery does not deplete below zero
        if self.energy < 0:
            self.energy = 0
            # Handle power outage scenario if necessary

    def get_energy(self):
        return self.energy

    def is_power_available(self, power_consumption, time_step):
        # Check if enough power is available for the next operation
        future_energy = self.energy - power_consumption * time_step
        return future_energy >= 0