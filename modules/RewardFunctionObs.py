import numpy as np

def calculate_reward(observer_satellite, pointing_accuracy):

    # # Calculate pointing accuracy reward
    # if pointing_accuracy is None:
    #     pointing_reward = 0  # Assign a default value or handle None case appropriately
    # else:
    #     pointing_reward = pointing_accuracy
    # Calculate pointing accuracy reward
    FoV = 1
    par_a = 1
    par_b = 1
    par_c = 1

    pointing_reward = pointing_accuracy

    # Calculate power availability reward
    power_reward = calculate_power_reward(observer_satellite)

    # Calculate availability reward
    availability_reward = calculate_availability_reward(observer_satellite)

    # Calculate instrumentation reward
    instrumentation_reward = calculate_instrumentation_reward(observer_satellite)

    # Calculate storage availability reward
    storage_reward = calculate_storage_reward(observer_satellite)

    # Sum up the rewards with appropriate weights if necessary
    if pointing_reward > FoV:
        total_reward=0
    else:
        total_reward = availability_reward * instrumentation_reward * (par_a*power_reward + par_b*storage_reward + par_c*(1-abs(pointing_reward/FoV)))
        # if total_reward>0:
        #     print(f"\t\t\t NEW RF: {observer_satellite.name}: {total_reward}") # add target name
    return total_reward

def calculate_power_reward(observer_satellite):
    epsys = observer_satellite.epsys
    power_availableRF = epsys['EnergyAvailable'] / epsys['EnergyStorage']
    return power_availableRF

def calculate_availability_reward(observer_satellite):
    availability = observer_satellite.availability
    availabilityRF = availability['availability']
    return availabilityRF

def calculate_instrumentation_reward(observer_satellite):
    instrumentation = observer_satellite.instrumentation
    instrumentationRF=instrumentation['cameraVIS']
    return instrumentationRF

def calculate_storage_reward(observer_satellite):
    # Calculate storage available for this satellite
    DataHand=observer_satellite.DataHand
    storage_availableRF = DataHand['StorageAvailable'] / DataHand['DataStorage']
    return storage_availableRF

def calculate_bid(observer_satellite, pointing_accuracy, task_meta=None, FoV: float = 1.0,
                  k_var: float = 2.0, t_exec:int=None, num_steps:int=None,
                  time_dependency:bool=False):
    """
    Resource-aware variable cost (€/m^2) derived directly from the same state
    that drives your Reward Function (power, storage, availability, pointing).

    Bid model:
        bid = C_fixed_sat + C_var_dynamic(t) * A

    where:
      - C_fixed_sat: per-satellite random fixed cost (kept as-is)
      - C_var_dynamic(t): baseline variable rate scaled by 'scarcity' of resources at the execution timestep
      - A: task area in m^2 (task_meta['area_m2'])

    Scaling logic (monotone inverse to RF):
      1) Compute RF at this timestep using your existing calculate_reward(...).
      2) Normalize RF to [0,1] with an upper bound of 3.0 (par_a + par_b + par_c in your RF).
      3) Inflate variable rate as resources get tight: C_var_dyn = C_var_base * [1 + k_var * (1 - RF_norm)]

    Notes:
      - Availability=0, or infeasible pointing, returns +inf (won’t bid).
      - C_var_base is GLOBAL (same for everyone) so only the fixed part remains random.
    """
    # --- Feasibility gates (keep protocol unchanged) ---
    if pointing_accuracy is None or pointing_accuracy > FoV:
        return float('inf')

    A = None
    if isinstance(task_meta, dict):
        A = task_meta.get('area_m2', None)
    if A is None:
        return float('inf')
    try:
        A = float(A)
    except Exception:
        return float('inf')
    if A <= 0.0:
        return float('inf')

    # If not available, don't bid
    avail = observer_satellite.availability.get('availability', 1)
    if avail <= 0:
        return float('inf')

    # --- Fixed cost: keep the per-satellite random you already assign in SatelliteClass ---
    C_fixed_default = 25.0
    C_fixed = float(getattr(observer_satellite, 'cost', {}).get('C_fixed', C_fixed_default))

    # --- GLOBAL baseline variable rate (€/m^2): same for all satellites ---
    C_var_base_m2 = 0.1

    # --- Compute RF at this timestep and turn it into a scarcity multiplier ---
    # Your RF upper bound with par_a=par_b=par_c=1 is ~3.0 when everything is perfect.
    RF_raw = calculate_reward(observer_satellite, pointing_accuracy)
    RF_norm = max(0.0, min(1.0, RF_raw / 3.0))  # clamp to [0,1]

    # k_var (default 2.0) sets how aggressively the variable €/m² rises as resources tighten.
    # k_var = 1: worst-case variable rate = 2× baseline
    # k_var = 3: worst-case variable rate = 4× baseline
    # Pick what separates “I’m flush with power & storage” from “I’m scraping the barrel.”
    # k_var controls how sensitive the variable price is to scarcity.
    # RF_norm=1.0 -> factor=1.0 (abundant resources); RF_norm=0.0 -> factor=1+k_var (scarce).
    scarcity_factor = 1.0 + k_var * (1.0 - RF_norm)

    C_var_dynamic_m2 = C_var_base_m2 * scarcity_factor

    # --- Time-dependent component (earlier execution ⇒ cheaper) ---
    C_time = 0.0
    if time_dependency:
        C_time_base = 10.0  # € — to tweak
        if isinstance(t_exec, (int, float)):
            if isinstance(num_steps, (int, float)) and num_steps and num_steps > 1:
                t_hat = max(0.0, min(1.0, float(t_exec) / float(num_steps - 1)))
            else:
                # fall-back: monotone in t_exec without needing horizon
                t_hat = float(t_exec) / (float(t_exec) + 1.0)
            C_time = C_time_base * t_hat

    # --- Final bid ---
    bid = C_fixed + C_time + C_var_dynamic_m2 * A
    return float(bid)
