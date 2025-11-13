from modules.SatelliteClass import Satellite, ObserverSatellite
from modules.OrbitPropagator import propagate_orbit, distance_between
from modules.Pointing_accuracy import evaluate_pointing_accuracy
from modules.RewardFunctionObs import calculate_reward, calculate_bid
from modules.CommSubsystemNEW import CommSubsystem
from modules.AttitudePropagatorWithEclipse import is_in_eclipse, sun_position
from modules.logger import best_two_from, print_bid_event
import math

def update_subsystems(satellite:Satellite, timestep:int, dt:float):

    # Orbit propagated in place
    pos1, _ = propagate_orbit(satellite, timestep*dt)

    # Determine if the satellite is in eclipse or sunlight
    if is_in_eclipse(satellite, sun_position(timestep * dt)):
        # Eclipse condition: No power generation, reduced power consumption
        #print("The satellite is in eclipse at time step", future_step)
        #print("future step is", future_step)  

        satellite.power_subsystem.generate_power(satellite, dt)  # Should generate 0 power
        reduced_power_consumption = satellite.epsys['EclipsePower']  # Define reduced power consumption rate
        satellite.power_subsystem.consume_power(reduced_power_consumption, dt)
    else:
        # Sunlit condition: Power generation, normal or increased power consumption
        #print("The satellite is in sunlight at time step", future_step) 
        #print("future step is", future_step)  
        satellite.power_subsystem.generate_power(satellite, dt)  # Generate power based on sun exposure
        normal_power_consumption = satellite.epsys['OperatingPower']  # Define normal power consumption rate
        satellite.power_subsystem.consume_power(normal_power_consumption, dt)
    #print("The energy available is", satellite.power_subsystem.get_energy())
    # Update energy available in satellite's epsys dictionary for reward calculations
    satellite.epsys['EnergyAvailable'] = satellite.power_subsystem.get_energy()

    # Reduce processing time for each satellite currenttly processing a task
    
    if satellite.processing_time > 0:
        satellite.processing_time -= dt
    if satellite.processing_time < 0:
        satellite.processing_time = 0     

    # Unflag comm
    satellite.flags['comm'] = 0
    
    return pos1    





def comm_conditions(satA:Satellite, satB:Satellite,timestep:int):
    # TODO: check comm conditions after consensus on lowest bid is reached - for second price
    """
    Conditions for two satellites to communicate: (should be improved) (some conditions may be put outside of this loop to speed up the code)
    0. One of the two satellites is a central node
    1. Effective datarate > 0
    2. The two satellites have a nonequal set of tasks, or they have equal set of tasks but different RF values for the tasks 
       NOTE: for now, we assume that communication happens only if the two satellites have enough datarate to share all tasks
    3. Both are not processsing a task and can communicate
    4. Additional condition, checked afterwards: the data matrix entry for this satellite combination 
       (eg. how much data the two satellites attempting to communicate can exchange) exceeds the data size
    
    Args:
        satA (Satellite): The first satellite.
        satB (Satellite): The second satellite.
        timestep (int): The current time step in the simulation.
    Returns:
        conditions (list): A list of boolean conditions that must be met for communication to occur.
    """
    # 0. Determine if one of the two satellites is a central node
    CN_present = (satA.commsys['band'] == 5 or satB.commsys['band'] == 5)

    # 1. Calculate the effective data rate using Shannon's formula
    dist = distance_between(satA, satB, timestep)
    data_comms = CommSubsystem()
    eff_datarate = data_comms.calculateEffectiveDataRate(dist)

    # 2. Determine point 2, necessity of communication

    # 2.1 determine if the two satellites have the same set of tasks - True if the two sets of tasks are equal and not empty
    equal_set = set(satA.tasks.keys()) == set(satB.tasks.keys())

    # 2.2 determine if the two satellites have the same RF values for the tasks
    if equal_set:
        def g(s, t, k, default=float('inf')):
            return s.tasks[t]['global'].get(k, default)

        equal_RF = all(
            satA.tasks[t]['global'].get('RF', 0) == satB.tasks[t]['global'].get('RF', 0)
            for t in satA.tasks
        )
        equal_bids = all(
            g(satA, t, 'lowest_bid') == g(satB, t, 'lowest_bid') and
            g(satA, t, 'second_lowest_bid') == g(satB, t, 'second_lowest_bid')
            for t in satA.tasks
        )
        equal_dict = equal_RF and equal_bids
    else:
        equal_dict = False

        # # If for some reason they have a different set of tasks, raise error:
        # if len(satA.tasks) != len(satB.tasks):
        #     raise ValueError("Dictionaries must have the same number of keys")
        # # If the sets are equal, check if the RF values are also equal
        # equal_dict = all(satA.tasks[task_id]['global']['RF'] == satB.tasks[task_id]['global']['RF'] for task_id in satA.tasks.keys())

    # 2.3 Check that at least one of the two satellites has a non-empty set of tasks
    if len(satA.tasks) > 0:
        non_empty = True
    elif len(satB.tasks) > 0:
        non_empty = True
    else:
        non_empty = False

    # 2.4 put it all together: satellites have a non-equal set of tasks or equal set of tasks but different RF values, and at least one of the two satellites has a non-empty set of tasks
    # OLD: knowledge_condition = (not equal_set or not equal_dict) and non_empty
    knowledge_condition = (not equal_set or not equal_dict) and non_empty
    # 3. Check processing time
    can_process = satA.processing_time == satB.processing_time == 0

    # Create set of conditions to check
    conditions = [  CN_present,
                    eff_datarate > 0,
                    knowledge_condition,
                    can_process]
    '''
    if  CN_present:
        print(f"Present timestep {k}. Comms between {satA.name} and {satB.name} (datarate, equal tasks, can process) : [{conditions[1]}, {conditions[2]}, {conditions[3]}]")
    '''

    return conditions, eff_datarate

def bidding_node(time_step, satA:ObserverSatellite, satB:ObserverSatellite,
                 tasks_known:set, tasks_unknown:set, best_reward_info:dict,
                 target_satellites:list, num_steps:int, max_distance:float,
                 k_var:float=2.0,
                 time_dependency:bool=False):

    """
    Sat A sends its tasks to Sat B, which updates its knowledge based on the received tasks. Sat A and B are processes in place.

    Args:
        time_step (int): The current time step in the simulation.
        satA (Satellite): The satellite sending its tasks.
        satB (Satellite): The satellite receiving tasks.
        tasks_known (list): List of task IDs known to satB.
        tasks_unknown (list): List of task IDs unknown to satB.
        best_reward_info (dict): Dictionary containing the best reward information for tasks.
        target_satellites (list): List of target satellites.
        num_steps (int): Total number of time steps in the simulation.
        max_distance (float): Maximum allowed distance for the task.
    Returns:
        best_reward_info (dict): Updated best reward information for tasks.
    """
   
    
    update_known_tasks(tasks_known, satA, satB)
    update_unknown_tasks(tasks_unknown, satA, satB)
    #print(f"\t {satA.name}: {[int(key) for key in satB.tasks.keys()]} -> {satB.name} ")


    # Evaluate the reward for satellite j for each of its new tasks starting from the current time step
    for task_id in tasks_unknown:  
        # Initialize temp subsystem
        # Input the same energy available to satellite at current timestep
        for future_step in range(time_step, num_steps):
            
            # Updated subsystems at future step

            # Process the task at the future time step
            process_task_at_timestep(task_id, satB, target_satellites,
                                     future_step, max_distance, k_var,
                                     num_steps=num_steps,
                                     time_dependency=time_dependency)

        # Process the new RF for the task
        best_reward_info = process_new_RF(task_id, satB, best_reward_info)
        best_reward_info = process_bid(task_id, satB, best_reward_info)

    return best_reward_info

def _top2_merge(A, B):
    cand = []
    # collect candidates (id, bid, time, RF) — skip non-finite bids
    for src in (A, B):
        for which in ("lowest", "second_lowest"):
            bid = src.get(f"{which}_bid", float("inf"))
            if math.isfinite(bid):
                cand.append({
                    "id":   src.get(f"{which}_id"),
                    "bid":  float(bid),
                    "time": src.get(f"{which}_time"),
                    "RF":   float(src.get(f"{which}_RF", 0.0)),
                })
    # deduplicate by id (keep best bid per id)
    by_id = {}
    for c in cand:
        if c["id"] is None:
            continue
        if c["id"] not in by_id or c["bid"] < by_id[c["id"]]["bid"]:
            by_id[c["id"]] = c
    cand = sorted(by_id.values(), key=lambda x: (x["bid"], x.get("time") or float("inf"), x["id"]))
    best = cand[0] if len(cand) >= 1 else None
    second = next((c for c in cand[1:] if best and c["id"] != best["id"]), None) if best else None
    return best, second



def update_known_tasks(tasks_known:set, satA:ObserverSatellite, satB:ObserverSatellite):
    """
    This function updates the RF values in satB for the tasks that both satellites already have in their dictionaries.
    For each already-known task, if the sender satellite (satA) has a higher RF value than the receiver satellite (satB),
    satB's task information is updated with the global information from satA.
    SatB and A are changed in place, and simulation-level knowledge is returned by the function.

    Args:
        tasks_known (set): Set of task IDs known to satB.
        tasks_unknown (set): Set of task IDs unknown to satB.
        satA (ObserverSatellite): The satellite sending its tasks.
        satB (ObserverSatellite): The satellite receiving tasks.

    Returns:
        None: The function updates the task information in place.
    """

    ## For tasks previously known to B, but with different best RF data:
    # Update max RF value and related variables 
    for task_id in tasks_known:
        A = satA.tasks[task_id]["global"]
        B = satB.tasks[task_id]["global"]
        best, second = _top2_merge(A, B)

        # apply merged view to BOTH (so info doesn’t ping-pong)
        for s in (satA, satB):
            g = s.tasks[task_id]["global"]
            # clear first
            g["lowest_id"] = g["second_lowest_id"] = None
            g["lowest_bid"] = g["second_lowest_bid"] = float("inf")
            g["lowest_time"] = g["second_lowest_time"] = None
            g["lowest_RF"] = g["second_lowest_RF"] = 0.0

            if best:
                g["lowest_id"] = best["id"]
                g["lowest_bid"] = best["bid"]
                g["lowest_time"] = best["time"]
                g["lowest_RF"] = best["RF"]
            if second:
                g["second_lowest_id"] = second["id"]
                g["second_lowest_bid"] = second["bid"]
                g["second_lowest_time"] = second["time"]
                g["second_lowest_RF"] = second["RF"]

            # update thinks_winner flag
            s.tasks[task_id]["thinks_winner"] = int(g.get("lowest_id") == s.name)

    return None



def update_unknown_tasks(tasks_unknown:set, satA:ObserverSatellite, satB:ObserverSatellite):

    """
    This function updates satellite B's task dictionary by adding the tasks previously unknown from satellite A.
    The local task information is set to default values, while the global task information is taken from satellite A.
    Satellite B is updated in place, and simulation-level knowledge is returned by the function.

    Args:
        tasks_unknown (set): Set of task IDs unknown to satB.
        satA (ObserverSatellite): The satellite sending its tasks.
        satB (ObserverSatellite): The satellite receiving tasks.
    Returns:
        None: The function updates the task information in place.
    """

    for task_id in tasks_unknown:
        satB.tasks[task_id] = {
            "local": {
                "RF": 0,
                "time": None,
                "bid": float("inf")
            },
            "global": {
                "RF": satA.tasks[task_id]["global"].get("RF", 0),
                "time": satA.tasks[task_id]["global"].get("time", 0),
                "id": satA.tasks[task_id]["global"].get("id", None),
                "lowest_bid": satA.tasks[task_id]["global"].get("lowest_bid", float("inf")),
                "second_lowest_bid": satA.tasks[task_id]["global"].get("second_lowest_bid", float("inf")),
                "second_lowest_time": satA.tasks[task_id]["global"].get("second_lowest_time", None),
                "second_lowest_RF": satA.tasks[task_id]["global"].get("second_lowest_RF", 0),
                "lowest_id": satA.tasks[task_id]["global"].get("lowest_id", None),
                "second_lowest_id": satA.tasks[task_id]["global"].get("second_lowest_id", None),
            },
            "meta": satA.tasks[task_id].get("meta", {}),
            "thinks_winner": 0
        }

    return


def process_task_at_timestep(task_id, satB:ObserverSatellite, target_satellites:list,
                             future_step:int, max_distance:float, k_var:float=2.0,
                             num_steps:int=None, time_dependency:bool=False):
    """
    This function processes the task at a given future time step for satellite B.
    It calculates the distance to the target satellite and evaluates the pointing accuracy.
    If the distance is within the maximum allowed distance, it calculates the reward value and updates the task information.

    Args:
        task_id (int): The ID of the task being processed.
        satB (ObserverSatellite): The satellite processing the task.
        target_satellites (list): List of target satellites.
        future_step (int): The future time step at which the task is being processed.
        max_distance (float): The maximum allowed distance for the task.
    
    Returns:
        None: The function updates the task information in place.
    """
    
    # target satellites are indexed by task_id as these should be initialised properly at the start
    future_distance = distance_between(satB, target_satellites[int(task_id)], future_step) / 1000
    if future_distance < max_distance:
        pointing_accuracy = evaluate_pointing_accuracy(satB, target_satellites[int(task_id)])
        reward_value = calculate_reward(satB, pointing_accuracy)

        try:
            bid_val = calculate_bid(
                satB,
                pointing_accuracy,
                task_meta=satB.tasks[task_id].get('meta', None),
                k_var=k_var,
                t_exec=future_step,
                num_steps=num_steps,
                time_dependency=time_dependency
            )
        except TypeError:
            # Back-compat with tests that monkeypatch a 3-arg calculate_bid(...)
            # (unit tests compatibility)
            bid_val = calculate_bid(
                satB,
                pointing_accuracy,
                task_meta=satB.tasks[task_id].get('meta', None)
            )

        if math.isfinite(bid_val) and bid_val < satB.tasks[task_id]['local']['bid']:
            # Save reward and time step - satellite level knowledge,
            satB.tasks[task_id]['local']['RF'] = reward_value
            satB.tasks[task_id]['local']['time'] = future_step
            satB.tasks[task_id]['local']['bid'] = bid_val
            print(
                f"\t\t\t New RF of {reward_value:.4f} computed at timestep {future_step}, "
                f"with associated (new lowest) bid of {satB.tasks[task_id]['local']['bid']:.4f} "
                f"for satellite {satB.name} (task {task_id})"
            )
    return None


def process_new_RF(task_id, satB:ObserverSatellite, best_reward_info:dict):

    """
    This function processes the new reward function (RF) for a task at a given future time step for satellite B.
    It updates the global best RF if satellite B has a higher RF than the current global best
    and records the reward value history.

    Args:
        task_id (int): The ID of the task being processed.
        satB (ObserverSatellite): The satellite processing the task.
        best_reward_info (dict): Dictionary containing the best reward information for tasks.
        reward_value_history (dict): Dictionary to store the history of reward values.
    
    Returns:
        best_reward_info (dict): Updated best reward information for tasks.
    """
    

    # Track the maximum reward and time step as far as satellite j is concerned
    if satB.tasks[task_id]['local']['RF'] > satB.tasks[task_id]['global']['RF']:
        satB.tasks[task_id]['global'].update(satB.tasks[task_id]['local']) 
        satB.tasks[task_id]['global']['id'] = satB.name
        # satB.tasks[task_id]['thinks_best'] = 1
        satB.flags['RF'] = 1                # Set the RF flag to 1, indicating that satB has at least one local RF != 0

        #print(f'\t\t{satB.name} thinks it has the best RF for task {task_id} with value {satB.tasks[task_id]["local"]["RF"]} at time step {satB.tasks[task_id]["local"]["time"]}')

        # Update the global best reward if satellite j has a higher reward - simulation level knowledge
        if satB.tasks[task_id]['global']['RF'] > best_reward_info[task_id]["RF"]:
            best_reward_info[task_id]["RF"] = satB.tasks[task_id]['global']['RF']
            best_reward_info[task_id]["id"] = satB.name
            best_reward_info[task_id]["time"] = satB.tasks[task_id]['global']['time']


    else:
        #print(f'\t\t{satB.name} does NOT think it has the best RF for task {task_id} with value {satB.tasks[task_id]["local"]["RF"]} at time step {satB.tasks[task_id]["local"]["time"]}')

        pass
    return best_reward_info

def _ensure_bid_fields(d: dict) -> None:
    d.setdefault("lowest_id", None)
    d.setdefault("lowest_bid", float("inf"))
    d.setdefault("lowest_time", None)
    d.setdefault("lowest_RF", 0.0)
    d.setdefault("second_lowest_id", None)
    d.setdefault("second_lowest_bid", float("inf"))
    d.setdefault("second_lowest_time", None)
    d.setdefault("second_lowest_RF", 0.0)

def _demote_lowest_to_second(d: dict) -> None:
    _ensure_bid_fields(d)
    if math.isfinite(d["lowest_bid"]):
        d["second_lowest_id"]   = d["lowest_id"]
        d["second_lowest_bid"]  = d["lowest_bid"]
        d["second_lowest_time"] = d["lowest_time"]
        d["second_lowest_RF"]   = d["lowest_RF"]
    else:
        d["second_lowest_id"]   = None
        d["second_lowest_bid"]  = float("inf")
        d["second_lowest_time"] = None
        d["second_lowest_RF"]   = 0.0

def process_bid(task_id: str, sat, best_reward_info: dict):
    """
    Mirror of process_new_RF, but for bids (lower is better).
    - Updates THIS satellite's global in-place (promote/demote).
    - Upserts sim-level bids list and keeps sim-level top-2 in sync.
    - Sets thinks_winner from THIS satellite's global.
    """
    # --- safety: never work with None ---
    if best_reward_info is None:
        best_reward_info = {}

    # ensure per-task dict exists
    bri = best_reward_info.setdefault(task_id, {})
    # ensure sim-level structures exist
    _ensure_bid_fields(bri)
    bids = bri.setdefault("bids", [])

    # read local values (cast to floats safely)
    local   = sat.tasks[task_id].setdefault("local", {})
    my_bid  = local.get("bid", float("inf"))
    my_RF   = local.get("RF", 0.0)
    my_time = local.get("time", None)

    try:
        my_bid = float(my_bid)
    except Exception:
        my_bid = float("inf")
    try:
        my_RF = float(my_RF)
    except Exception:
        my_RF = 0.0

    # ---- upsert into sim-level bids list (used by summaries) ----
    # remove any previous entry for me
    bids[:] = [b for b in bids if b.get("id") != sat.name]
    # add only if finite + meaningful
    if math.isfinite(my_bid) and my_RF > 0.0:
        bids.append({"id": sat.name, "bid": my_bid, "RF": my_RF, "time": my_time})
        # deterministic sort & tie-breaker
        bids.sort(key=lambda x: (float(x["bid"]),
                                 float(x.get("time", float("inf"))),
                                 str(x.get("id", ""))))

    # ---- update THIS satellite's global view (promote/demote) ----
    g = sat.tasks[task_id].setdefault("global", {})
    _ensure_bid_fields(g)

    if math.isfinite(my_bid):
        if my_bid < g["lowest_bid"]:
            _demote_lowest_to_second(g)
            g["lowest_id"]   = sat.name
            g["lowest_bid"]  = my_bid
            g["lowest_time"] = my_time
            g["lowest_RF"]   = my_RF
        elif my_bid < g["second_lowest_bid"] and sat.name != g["lowest_id"]:
            g["second_lowest_id"]   = sat.name
            g["second_lowest_bid"]  = my_bid
            g["second_lowest_time"] = my_time
            g["second_lowest_RF"]   = my_RF

    # local belief flag (based on HIS global)
    sat.tasks[task_id]["thinks_winner"] = int(g.get("lowest_id") == sat.name)

    # ---- keep sim-level top-2 fields in sync with the bids list ----
    if bids:
        L = bids[0]
        bri["lowest_id"]   = L["id"]
        bri["lowest_bid"]  = float(L["bid"])
        bri["lowest_time"] = L.get("time")
        bri["lowest_RF"]   = float(L.get("RF", 0.0))
        if len(bids) >= 2:
            S = bids[1]
            bri["second_lowest_id"]   = S["id"]
            bri["second_lowest_bid"]  = float(S["bid"])
            bri["second_lowest_time"] = S.get("time")
            bri["second_lowest_RF"]   = float(S.get("RF", 0.0))
        else:
            bri["second_lowest_id"]   = None
            bri["second_lowest_bid"]  = float("inf")
            bri["second_lowest_time"] = None
            bri["second_lowest_RF"]   = 0.0

    # always return the dict to avoid passing None to the next call
    return best_reward_info