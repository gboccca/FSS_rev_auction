import sys, os, time, copy, pickle
import numpy as np
import pandas as pd
from datetime import datetime
import math
import random as _py_random
from scipy.spatial import cKDTree
# --- Parallel runner glue (add this) ---
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class SimConfig:
    run_name: str
    verbosity: bool
    type: str
    num_tasks: int
    num_satellites: int
    fraction_CNs: float
    dt: int = 1
    duration: int = 6*3600
    k_var: float = 2.0
    constellation: str = "walker_delta_10x50"
    preset_size: bool = True
    time_dependency: bool = False
    omni_fraction_CNs: float = 0.0
    seed: Optional[int] = None
    randomize_constellation: bool = False
    out_dir: str = "outputs"  # <— where to save artifacts

def run_single_sim(cfg: SimConfig) -> Dict[str, Any]:
    """
    Module-level function so it's PICKLABLE by multiprocessing.
    Runs one simulation and returns a compact summary dict.
    """
    sim = Sim(
        run_name=cfg.run_name,
        verbosity=cfg.verbosity,
        type=cfg.type,
        num_tasks=cfg.num_tasks,
        num_satellites=cfg.num_satellites,
        fraction_CNs=cfg.fraction_CNs,
        dt=cfg.dt,
        duration=cfg.duration,
        k_var=cfg.k_var,
        constellation=cfg.constellation,
        preset_size=cfg.preset_size,
        time_dependency=cfg.time_dependency,
        omni_fraction_CNs=cfg.omni_fraction_CNs,
        seed=cfg.seed,
        randomize_constellation=cfg.randomize_constellation,
    )
    summary = sim.run_full(outputfolder=cfg.out_dir, savedata=True, savehistory=False)
    # Ensure returned dict is JSON-serializable and small:
    return {
        "run_name": cfg.run_name,
        "elapsed_s": summary.get("elapsed_s"),
        "timesteps": summary.get("timesteps"),
        "saved_file": summary.get("saved_file"),
    }

# Add the parent directory to the system path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.SatelliteClass import ObserverSatellite, TargetSatellite
from modules.OpticPayload import OpticPayload
from modules.PowerSubsystemV2 import PowerSubsystem
from modules.bidding_logic import bidding_node, comm_conditions, update_subsystems, process_task_at_timestep, process_new_RF, process_bid
from modules.logger import print_timestep_summary, _list_or_default, _ensure_ts



class Sim:

    def __init__(self, run_name:str, verbosity:bool, type:str,
                 num_tasks:int, num_satellites:int, fraction_CNs:float,
                 dt=1, duration=6*3600, k_var=2.0,
                 constellation:str="walker_delta_10x50",
                 preset_size:bool=True,
                 time_dependency:bool=False,
                 omni_fraction_CNs:float=0.0,
                 seed:int|None=None,
                 randomize_constellation:bool=False
                 ):
        """
        Initializes the simulation with the given parameters.
        :param run_name: Name of the simulation run.
        :param type: Type of simulation (e.g., 'centralized', 'partially_centralized').
        :param num_tasks: Number of tasks to be processed in the simulation.
        :param num_satellites: Number of satellites in the simulation.
        :param fraction_CNs: Fraction of satellites that are Central Nodes (CNs).
        :param dt: Time step for the simulation in seconds.
        :param duration: Total duration of the simulation in seconds.
        """

        ### Initialize simulation parameters
        self.start_time = time.time()
        self.name = run_name
        self.verbosity = verbosity
        self.type = type
        self.num_tasks = num_tasks
        self.num_satellites = num_satellites
        self.fraction_CNs = fraction_CNs
        self.dt = dt
        self.duration = duration
        self.k_var = k_var
        self.time_dependency = bool(time_dependency)
        self.num_steps = int(duration / dt)
        self.breaker = False
        self.k = 0 # Current time step in the simulation
        self.state_str = ""  # String to hold the current state of the simulation

        self.unsolved_tasks = set()  # tasks that triggered "full-knowledge but <2 bids"

        # this is to make costs, tasks metadata and subsystems initialisation
        # reproducible when seed is provided in Sim() instantiation
        # otherwise, it stays stochastic
        if seed is not None:
            np.random.seed(int(seed))
            _py_random.seed(int(seed))
            print(f"[INIT] RNG seeded with seed={seed}")
        else:
            print(f"[INIT] RNG not seeded (non-reproducible run).")

        self.seed = int(seed) if seed is not None else None
        self.randomize_constellation = bool(randomize_constellation)

        # --- Constellation presets (rough, but useful) ---
        # If a preset sets planes/sats_per_plane, we use them and (optionally) override user num_satellites.
        PRESETS = {
            # Generic Walker (delta) examples
            "walker_delta_10x50": {
                "mode": "Walker",
                "planes": 10,
                "sats_per_plane": 50,
                "semimajoraxis_km": 6921,  # ~550 km alt
                "inclination_deg": 53,
                "walker_cfg": {"F": 1}  # phasing
            },
            "walker_delta_20x30": {
                "mode": "Walker",
                "planes": 20,
                "sats_per_plane": 30,
                "semimajoraxis_km": 6921,
                "inclination_deg": 53,
                "walker_cfg": {"F": 2}
            },
            # "Star" flavor
            "walker_star_20x30": {
                "mode": "Walker",
                "planes": 20,
                "sats_per_plane": 30,
                "semimajoraxis_km": 6921,
                "inclination_deg": 53,
                "walker_cfg": {"F": 7}
            },
            # Approx Starlink “shell 1”
            "starlink_shell_1": {
                "mode": "Walker",
                "planes": 72,
                "sats_per_plane": 22,
                "semimajoraxis_km": 6921,
                "inclination_deg": 53.0,
                "walker_cfg": {"F": 17}
            },
            # Approx Planet “SuperDove”-like SSO cluster (very rough)
            "planet_sso": {
                "mode": "Walker",  # still use Walker distribution machinery
                "planes": 12,
                "sats_per_plane": 12,
                "semimajoraxis_km": 6853,  # ~475 km alt
                "inclination_deg": 98.0,  # SSO-ish
                "walker_cfg": {"F": 0}
            },
        }

        cfg = PRESETS.get(constellation, None)
        if cfg is None:
            raise ValueError(f"Unknown constellation preset '{constellation}'. "
                             f"Valid keys: {list(PRESETS.keys())}")

        self.constellation = constellation
        self.orbit_type = cfg["mode"]

        # Honor fixed-size presets by overriding num_satellites
        if preset_size:
            # Use the constellation’s native size
            self.num_planes = int(cfg["planes"])
            self.num_satellites_per_plane = int(cfg["sats_per_plane"])
            self.num_satellites = self.num_planes * self.num_satellites_per_plane
        else:
            # Use user-specified num_satellites, distributed across preset planes
            self.num_planes = int(cfg["planes"])
            self.num_satellites_per_plane = max(1, int(round(num_satellites / self.num_planes)))
            self.num_satellites = num_satellites


        # Store for passing down to SatelliteClass
        self.constellation_kwargs = {
            "semimajoraxis_km": cfg.get("semimajoraxis_km"),
            "inclination_deg": cfg.get("inclination_deg"),
            "walker_cfg": cfg.get("walker_cfg", {}),
        }

        ## Orbit Parameters
        # self.orbit_type = 'Walker'
        # self.num_planes = 10
        # self.num_satellites_per_plane = int(num_satellites / self.num_planes)

        ## Satellite/Subsystem Parameters
        self.processing_time = 1
        self.Data_Size = 1680*self.num_tasks
        self.optic_payload = OpticPayload()

        # TODO: check if it was good to multiply by 1.5
        self.max_distance = self.optic_payload.dist_detect()/1000 * 1.5

        ## Initialize Satellites

        # Target Satellites
        self.target_satellites = [TargetSatellite(
            mode = self.orbit_type,
            num_planes= self.num_planes,
            num_satellites_per_plane= self.num_satellites_per_plane,
            index=i,
            name=f'Target-{i+1}',
            category='target',
            total_satellites=self.num_satellites
        ) for i in range(num_tasks)]
        
        # Differentiate target satellites orbit by index - equal spacing in true anomaly (added randomness)
        # after creating self.target_satellites
        if self.randomize_constellation:
            for tgt in self.target_satellites:
                tgt.orbit['raan'] = float(np.random.uniform(0.0, 360.0))
                tgt.orbit['true_anomaly'] = float(np.random.uniform(0.0, 360.0))
        else:
            for i, tgt in enumerate(self.target_satellites):
                tgt.orbit['true_anomaly'] = i * (360 / self.num_tasks)

        # Configuration creation. Possibility to make it randomised is added. If randomize_constellation:
        # - Rotate the whole Walker grid by random global offsets in RAAN and true anomaly
        # - Rotate the index→slot mapping by a random cyclic shift so observers don’t always take the same plane/slot
        indices = list(range(self.num_satellites))
        #
        # # this preserves the Walker math in SatelliteClass but changes which sat occupies each plane/slot
        # if self.randomize_constellation:
        #     start = int(np.random.randint(0, self.num_satellites))
        #     indices = [(start + i) % self.num_satellites for i in range(self.num_satellites)]
        
        # Observer Satellites
        self.satellites = [ObserverSatellite(
            mode=self.orbit_type,
            num_planes=self.num_planes,
            num_satellites_per_plane=self.num_satellites_per_plane,
            index=idx,
            name=f'Observer-{i + 1}',
            category='observer',
            total_satellites=self.num_satellites,
            **self.constellation_kwargs
        ) for i, idx in enumerate(indices)]

        # Persist (plane, slot) per satellite for robust reshuffling
        for i, sat in enumerate(self.satellites):
            sat.orbit['plane_idx'] = i // self.num_satellites_per_plane
            sat.orbit['slot_idx'] = i % self.num_satellites_per_plane

        # Apply the Walker-preserving reshuffle
        if self.randomize_constellation:
            self._apply_walker_reshuffle()

        # preserves preset SMA and inclination, but randomizes the constellation’s inertial orientation and along-track placement.
        # Walker elements come from SatelliteClass.__init__; here its just post-adjusting angles
        # if self.randomize_constellation:
        #     raan_offset = float(np.random.uniform(0.0, 360.0))
        #     ta_offset = float(np.random.uniform(0.0, 360.0))
        #     for sat in self.satellites:
        #         sat.orbit['raan'] = (sat.orbit['raan'] + raan_offset) % 360.0
        #         sat.orbit['true_anomaly'] = (sat.orbit['true_anomaly'] + ta_offset) % 360.0

        ## Initiallize power subsystems
        for satellite in self.satellites:
            satellite.power_subsystem = PowerSubsystem(satellite)

        self.omni_fraction_CNs = float(max(0.0, min(1.0, omni_fraction_CNs)))

        ### Initialize Federation (compute CNs AFTER preset, OUTSIDE the loop)
        if self.type == 'centralized':
            self.num_central_nodes = self.num_satellites
        elif self.type == 'partially_centralized':
            self.num_central_nodes = int(self.num_satellites * self.fraction_CNs)
        else:
            raise ValueError("Invalid simulation type. Choose 'centralized' or 'partially_centralized'.")

        # Initialize Central Nodes (CNs) based on simulation type
        step_size = max(1, self.num_satellites // max(1, self.num_central_nodes))
        if self.randomize_constellation:
            selected_indices = list(np.random.choice(range(self.num_satellites),
                                                     size=self.num_central_nodes, replace=False))
        else:
            selected_indices = [i * step_size for i in range(self.num_central_nodes)]

        # selected_indices = np.random.choice(range(self.num_satellites), size=self.num_central_nodes, replace=False) # uncomment for random selection

        # Assign Central Nodes
        self.satellites[0].flag = 1
        self.satellites[0].commsys['band'] = 5

        for idx in selected_indices:
            satellite = self.satellites[idx]
            satellite.commsys['band'] = 5
            satellite.flags['central'] = 1  # Set the central flag to 1 for Central Nodes
            satnumber = satellite.name.split('-')[-1]
            satellite.name = 'CentralNode-' + str(satnumber)  # Rename the satellite to CentralNode-X

        # --- NEW: choose the omniscient subset among the CNs
        omni_count = int(round(self.num_central_nodes * self.omni_fraction_CNs))
        omni_count = max(0, min(self.num_central_nodes, omni_count))
        self._omni_cn_indices = selected_indices[:omni_count]
        self._omni_cns = [self.satellites[i] for i in self._omni_cn_indices]

        ### Initialize Data Structures
        self.breaker = False
        self.positions = np.zeros((self.num_satellites, 3))
        self.data_matrix = np.zeros((self.num_satellites, self.num_satellites), dtype=int)

        # OPTIMIZATION: Create satellite lookup dictionary for O(1) access
        self.satellite_by_name = {sat.name: sat for sat in self.satellites}

        # OPTIMIZATION: Maximum communication range (km) - from CommSubsystem sensitivity
        # Based on CommSubsystemNEW parameters: SENSITIVITY = -151 dBW, can calculate max range
        # Approximate max range ~2400 km for UHF band
        self.max_comm_range_km = 2500  # km, conservative estimate

        #data_matrix_acc = np.zeros((num_satellites, num_satellites), dtype=int)
        self.adjacency_matrices = []  

        self.best_reward_info = {}
        self.bidding_info = {}
        for i in range(num_tasks):
            # best_reward_info per task: single source of truth for current top-2 bids
            self.best_reward_info[str(i)] = {
                "RF": 0, "id": None, "time": 0,  # legacy RF tracker (kept but unused for winners)
                "bids": [],

                # winner (lowest bid)
                "winner_id": None,
                "winner_bid": float("inf"),
                "winner_RF": 0,
                "winner_time": 0,

                # runner-up (second-lowest bid)
                "runner_up_id": None,
                "runner_up_bid": float("inf"),
                "runner_up_RF": 0,
                "runner_up_time": 0,

                # 2nd price to be paid by the winner
                "price_paid_2nd": float("inf"),
            }

            # time-series we will export to Excel (ONLY these fields)
            self.bidding_info[str(i)] = {
                "timestep": [],
                "RF_time": [],  # time at which winner_RF was computed
                "knowledge": [],
                "thinks_winner": [],

                "winner_id": [],
                "winner_bid": [],
                "winner_RF": [],

                "runner_up_id": [],
                "runner_up_bid": [],
                "runner_up_RF": [],

                "price_paid_2nd": [],
            }

        print(f"Simulation initialized with {self.num_tasks} tasks, "
              f"{self.num_satellites} satellites, and {self.num_central_nodes} central nodes.")
        print(f"Omniscient CNs at init: {omni_count}/{self.num_central_nodes} "
              f"({self.omni_fraction_CNs:.0%}).")

        print(f"Simulation parameters: k_var = {self.k_var}")

        print(f"[INIT] Constellation preset: {self.constellation} | "
              f"P={self.num_planes}, S/P={self.num_satellites_per_plane}, "
              f"i={self.constellation_kwargs.get('inclination_deg')}°, "
              f"SMA={self.constellation_kwargs.get('semimajoraxis_km')} km, "
              f"F={self.constellation_kwargs.get('walker_cfg', {}).get('F')}")

    def _apply_walker_reshuffle(self):
        """
        Preserve the preset Walker-like structure (equal RAAN spacing, sats/plane)
        but reshuffle: global RAAN rotation, inter-plane Walker phasing F,
        per-plane along-track offsets, and a tiny per-sat jitter.
        """
        import numpy as np

        P = int(self.num_planes)
        S_per_plane = int(self.num_satellites_per_plane)

        if P <= 0 or S_per_plane <= 0:
            return  # nothing to do

        # Draw random parameters (seed control already handled in Sim __init__)
        phi = float(np.random.uniform(0.0, 360.0))  # global RAAN rotation
        F = int(np.random.randint(0, S_per_plane))  # Walker phasing per run
        plane_phase = [float(np.random.uniform(0.0, 360.0 / S_per_plane))
                       for _ in range(P)]  # per-plane along-track
        jitter_sigma = 1.0  # 1° std tiny jitter

        # If you don't store plane/slot, infer them from index order
        # (this assumes you created sats plane-by-plane, slot-by-slot)
        for idx, sat in enumerate(self.satellites):
            plane_idx = sat.orbit.get('plane_idx')
            slot_idx = sat.orbit.get('slot_idx')

            if plane_idx is None or slot_idx is None:
                plane_idx = idx // S_per_plane
                slot_idx = idx % S_per_plane

            # Ensure plane_idx doesn't exceed bounds (can happen when preset_size=False)
            plane_idx = min(plane_idx, P - 1)

            # RAAN: equally spaced by 360/P, plus global rotation
            raan_i = (phi + (360.0 / P) * plane_idx) % 360.0

            # Along-track base index with Walker phasing F * plane_idx
            pos_idx = (slot_idx + F * plane_idx) % S_per_plane
            base_ta = pos_idx * (360.0 / S_per_plane)

            # Add per-plane phase and tiny jitter
            ta = base_ta + plane_phase[plane_idx]
            if jitter_sigma > 0.0:
                ta = (ta + float(np.random.normal(0.0, jitter_sigma))) % 360.0

            # Write back
            sat.orbit['raan'] = float(raan_i)
            sat.orbit['true_anomaly'] = float(ta)

            # (Optional but handy for debugging/analysis)
            sat.orbit['plane_idx'] = int(plane_idx)
            sat.orbit['slot_idx'] = int(slot_idx)
            sat.orbit['walker_F'] = int(F)

    def first_step(self):
        """
        Runs the first step of the simulation, initializing tasks inside the initial satellite.
        Only call this function once, after the Sim() object is created.
        NEW:
        If omni_fraction_CNs > 0, seed that fraction of CNs with all tasks at t=0,
        pre-compute their local RF & bid over the horizon, and merge their
        lowest/second-lowest among themselves. Otherwise, fall back to the legacy
        single-initial-observer seeding.
        """

        # --- NEW BRANCH: partial omniscience for a subset of CNs ---
        if getattr(self, "_omni_cns", None) and len(self._omni_cns) > 0:
            # one shared meta per task (so bids are comparable)
            task_metas = {}
            for task_id in range(self.num_tasks):
                area_m2 = float(np.random.uniform(10, 100))
                pixels_requested = int(np.random.randint(1080, 4000))
                task_metas[str(task_id)] = {
                    'area_m2': area_m2,
                    'megapixels': pixels_requested ** 2 / 1e6,
                    '_pixels_side': pixels_requested,  # for pretty logs
                }

            # initialize tasks on each omniscient CN
            for cn in self._omni_cns:
                cn.tasks.clear()
                for tid, meta in task_metas.items():
                    cn.tasks[tid] = {
                        'global': {
                            'RF': 0, 'id': None, 'time': 0,
                            'lowest_id': None, 'second_lowest_id': None,
                            'lowest_bid': float('inf'), 'second_lowest_bid': float('inf'),
                            'lowest_time': None, 'second_lowest_time': None,
                            'lowest_RF': 0.0, 'second_lowest_RF': 0.0,
                        },
                        'local': {'RF': 0, 'time': 0, 'bid': float('inf')},
                        'meta': meta,
                        'thinks_winner': 0,
                    }

            # each omniscient CN computes its own best RF & bid over the horizon
            for cn in self._omni_cns:
                for tid in list(cn.tasks.keys()):
                    for future_step in range(self.num_steps):
                        _ = update_subsystems(cn, future_step, self.dt)
                        process_task_at_timestep(
                            tid, cn, self.target_satellites,
                            future_step, self.max_distance, self.k_var,
                            num_steps=self.num_steps,
                            time_dependency=self.time_dependency
                        )
                    # publish CN's local best to sim-level & compute bid
                    self.best_reward_info = process_new_RF(tid, cn, self.best_reward_info)
                    self.best_reward_info = process_bid(tid, cn, self.best_reward_info)
                # --- Debug: print each omniscient CN’s best RF and bid
                print("\n[INIT] Omniscient CNs initial bids:")
                for tid, tdict in cn.tasks.items():
                    rf = tdict['local']['RF']
                    bid = tdict['local']['bid']
                    t = tdict['local']['time']
                    print(f"  {cn.name:15s} | Task {tid} | RF={rf:.4f} | Bid={bid:.4f} | t={t}")

            # merge top-2 across the omniscient set so they all start synchronized
            if len(self._omni_cns) > 1:
                # pairwise merge using your _top2_merge logic via update_known_tasks
                all_tids = set(task_metas.keys())
                from modules.bidding_logic import update_known_tasks  # local import
                for i in range(len(self._omni_cns)):
                    for j in range(i + 1, len(self._omni_cns)):
                        update_known_tasks(all_tids, self._omni_cns[i], self._omni_cns[j])

            # friendly logs
            for tid, meta in task_metas.items():
                any_cn = self._omni_cns[0]
                g = any_cn.tasks[tid]['global']
                print(
                    f"[INIT-OMNI] Task {tid}: seeded {len(self._omni_cns)} CNs with "
                    f"{meta['area_m2']:.2f}m² / {int(meta['_pixels_side'])}px "
                    f"({meta['area_m2'] / meta['_pixels_side']:.3f} m/px). "
                    f"Start lowest: {g.get('lowest_id')} | bid={g.get('lowest_bid', float('inf')):.4f} "
                    f"| RF={g.get('lowest_RF', 0.0):.4f} | t={g.get('lowest_time')}"
                )

            print('Completed first step (partial-omniscient CN seeding)')
            return

        # --- FALLBACK: your existing single initial observer path (unchanged) ---
        initial_observer = next(sat for sat in self.satellites if sat.flag == 1)

        # --- BEGIN: task meta randomization (add this before building the dict) ---
        area_m2 = float(np.random.uniform(10, 100))  # m2, square eqv: [1x1]m - [5x10]m
        pixels_requested = int(np.random.randint(1080, 4000))  # pixels per side
        # --- END ---

        # Initialize list of tasks in initial observer
        for task_id in range(self.num_tasks):
            initial_observer.tasks[str(task_id)] = {

                'global': { 'RF': 0,
                            'id': None,
                            'lowest_id': None,
                            'second_lowest_id': None,
                            'lowest_bid': float("inf"),
                            'second_lowest_bid': float("inf"),
                            'second_lowest_time': 0,            # TODO change into lowest_time
                            'second_lowest_RF': 0
                },
                'local': { 'RF': 0,
                           'time': 0,
                           'bid': float('inf')
                },

                # “resolution” can be computed as m/px (the lower value the highest res)
                'meta': {
                    'area_m2': area_m2,  # characteristic task size for imaging (1-50)
                    'megapixels': pixels_requested ** 2 / 1e6,  # Mpx (1.16-64)
                },

                'thinks_winner': 0,
            }

        # Iterate over all tasks assigned to the initial observer
        # note: this has too many inner loops, for every task each future step is reevaluated but there is no need to do that. 
        # Cannot change this: if you pull update_subsystems out of the loop: future step is not defined
        for task_id in initial_observer.tasks.keys():
            for future_step in range(self.num_steps):

                #### Update initial observer subsystems, in place
                pos = update_subsystems(initial_observer, future_step, self.dt)

                ## Evaluate RF for the initial observer
                process_task_at_timestep(task_id, initial_observer, self.target_satellites,
                                         future_step, self.max_distance,
                                         self.k_var,
                                         num_steps = self.num_steps,
                                         time_dependency = self.time_dependency
                                         )

            ## Update simulation-level knowledge
            self.best_reward_info = process_new_RF(task_id, initial_observer, self.best_reward_info)
            self.best_reward_info = process_bid(task_id, initial_observer, self.best_reward_info)

            print(
                f"[INIT] Task {task_id}: seeded first observer {initial_observer.name} "
                f"with an imaging requiring a resolution of {area_m2:.2f}m/{pixels_requested}px={area_m2/pixels_requested:.3f}m/px."
                f"For this task, {initial_observer.name} has RF={self.best_reward_info[str(task_id)]['RF']:.4f},\n "
                f"and is bidding bid={self.best_reward_info[str(task_id)].get('lowest_bid', float('nan')):.4f}"
            )

        print('Completed first step')


    def step(self):

        """
        Runs a single step of the simulation for all satellites.
        This method updates all satellites subsystem at the self.k timestep, processes bidding requests of all possible satellite pairs, and updates the simulation state.
    
        
        Returns:
            None
        
        """



        print(f"\n\n=================== Timestep {self.k}/{self.num_steps} ===================")

        # Update satellite subsystems and orbits, in place
        for i in range(len(self.satellites)):
            satellite = self.satellites[i]
            pos = update_subsystems(satellite, self.k, self.dt)
            # convert to km
            pos = np.array(pos) / 1000  # Convert position to kilometers
            self.positions[i] = pos  # Store the updated position in the positions array
            
        print(f"Updated subsystems for all satellites at time step {self.k}\n")

        # OPTIMIZATION: Use spatial indexing to find only nearby satellite pairs
        # Build KD-tree from satellite positions (already in km from line 625)
        tree = cKDTree(self.positions)

        # Query pairs within max communication range
        # query_pairs returns a set of (i, j) tuples where i < j
        satellite_pairs = tree.query_pairs(r=self.max_comm_range_km)

        print(f"Spatial indexing: checking {len(satellite_pairs)} pairs (out of {len(self.satellites)*(len(self.satellites)-1)//2} total) within {self.max_comm_range_km} km range\n") if self.verbosity else None

        # For each satellite pair within communication range, check if they can communicate
        for i, j in satellite_pairs:
            satA = self.satellites[i]
            satB = self.satellites[j]

            #print('now processing: ', satA.name, '->', satB.name) if self.verbosity else None

            conditions, eff_datarate = comm_conditions(satA, satB, self.k)

            if all(conditions):

                # Satellite B and A can exchange data: how much data?
                self.data_matrix[i][j] = self.data_matrix[j][i] = self.data_matrix[i][j] + (eff_datarate * self.dt)
                satA.flags['comm'] = 1
                satB.flags['comm'] = 1

                # If the data matrix entry for this satellite combination (eg. how much data the two satellites attempting to communicate can exchange) exceeds the data size, flag the second satellite (eg. exchange the tasks)
                # TO CHANGE: ALLOW TASKS TO BE EXCHANGED INDIVIDUALLY, NOT AS A BIG DATA PACKET
                if self.data_matrix[i][j] > self.Data_Size:


                    # Set the processing time for satellite B to the current processing time expected to calculate a single RFmax
                    # TO CHANGE: UPDATE PROCESSING TIME BASED ON THE NUMBER OF NEW TASKS AND THEIR COMPLEXITY
                    # FIX: Set processing time when a satellite is processing a new task -
                    satB.processing_time = self.processing_time


                    print(f'{satA.name} and {satB.name} will communicate with each other') if self.verbosity else None
                    ### Satellite j is receiving all the tasks from i

                    # OPTIMIZATION: Compute task key sets once to avoid redundant set() calls
                    keysA = set(satA.tasks.keys())
                    keysB = set(satB.tasks.keys())
                    tasks_known = keysA & keysB         # Identify tasks that both satellites know
                    tasks_unknown_to_B = keysA - keysB  # Identify tasks that were previously unknown to satellite B
                    tasks_unknown_to_A = keysB - keysA  # Identify tasks that were previously unknown to satellite A

                    print(f"Tasks known to both satellites: {tasks_known}") if self.verbosity else None
                    print(f"Tasks unknown to satellite {satB.name}: {tasks_unknown_to_B}") if self.verbosity else None
                    print(f"Tasks unknown to satellite {satA.name}: {tasks_unknown_to_A}") if self.verbosity else None

                    # Now processing satellite j
                    print(f"\tNow processing: {satA.name} -> {satB.name}") if self.verbosity else None
                    self.best_reward_info = bidding_node(
                        self.k,
                        satA,
                        satB,
                        tasks_known,
                        tasks_unknown_to_B,
                        self.best_reward_info,
                        self.target_satellites,
                        self.num_steps,
                        self.max_distance,
                        self.k_var,
                        time_dependency=self.time_dependency
                    )
                    satB.flags['tasks'] = 1

                    print(f"\tNow processing: {satB.name} -> {satA.name}") if self.verbosity else None

                    # Now processing satellite i
                    self.best_reward_info = bidding_node(
                        self.k,
                        satB,
                        satA,
                        tasks_known,
                        tasks_unknown_to_A,
                        self.best_reward_info,
                        self.target_satellites,
                        self.num_steps,
                        self.max_distance,
                        self.k_var,
                        time_dependency=self.time_dependency
                    )
                    satA.flags['tasks'] = 1

                    print("\n") if self.verbosity else None

            else:
                if eff_datarate <= 0:
                    self.data_matrix[i][j] = self.data_matrix[j][i] = 0
                #self.data_matrix[i][j] = self.data_matrix[j][i] = 0


                #print(f'{satellites[i].name} and {satellites[j].name} will NOT communicate with each other\n')
        #### Collect bidding data and update dictionary at current timestep
        for task_id in range(self.num_tasks):
            task_key = str(task_id)

            bi = self.bidding_info[task_key]
            bi["timestep"].append(self.k)

            # initialize counters for this timestep
            bi["knowledge"].append(0)
            bi["thinks_winner"].append(0) # TODO count for this task at this timestep

            # count how many satellites know the task, and how many think they're the winner
            for sat in self.satellites:
                if task_key in sat.tasks:
                    bi["knowledge"][self.k] += 1
                    if sat.tasks[task_key]['thinks_winner'] == 1:
                        bi["thinks_winner"][self.k] += 1

        ### Create a string with the current state of the simulation
        temp_str = f"At timestep {self.k}, the bidding info is:\n"

        for task_id in self.bidding_info.keys():
            bi = self.bidding_info[task_id]
            temp_str += (
                f'\tTask {task_id}: \n'
            )
            temp_str += (
                f'\t\tKnowledge: {bi["knowledge"][self.k]} satellites know this task \n'
            )
            temp_str += (
                f'\t\tThinks winner: {bi["thinks_winner"][self.k]} satellites think they have the lowest bid\n'
            )
            # Optional: if you want to show the current winner snapshot in this section too:
            # temp_str += (f'\t\tWinner: {bi["winner_id"][self.k]}, bid={bi["winner_bid"][self.k]}, RF={bi["winner_RF"][self.k]}, t={bi["RF_time"][self.k]}\n')
            # temp_str += (f'\t\tRunner-up: {bi["runner_up_id"][self.k]}, bid={bi["runner_up_bid"][self.k]}, RF={bi["runner_up_RF"][self.k]}, price_2nd={bi["price_paid_2nd"][self.k]}\n')

        # Print satellites that think they have the best RF for each task
        temp_str += (f'\nAt timestep {self.k}, satellites that think they have the lowest bid for each task:\n')
        for task_id in self.bidding_info.keys():
            temp_str += (f'\tTask {task_id}: \n')
            for sat in self.satellites:
                if task_id in sat.tasks and sat.tasks[task_id]['thinks_winner'] == 1:
                    local_bid = sat.tasks[task_id]['local'].get('bid', float('inf'))
                    global_low = sat.tasks[task_id]['global'].get('lowest_bid', float('inf'))
                    global_2nd = sat.tasks[task_id]['global'].get('second_lowest_bid', float('inf'))
                    temp_str += (
                        f'\t\tSatellite {sat.name}: '
                        f'local_bid={local_bid:.4f}, '
                        f'global_lowest={global_low:.4f}, '
                        f'global_second={global_2nd:.4f}\n'
                    )

        temp_str += 'End of current state report.'
        print(temp_str)

        self.state_str = temp_str  # Store the current state string for later use

        # ---- concise per-task end-of-timestep summaries and logging for save_data() ----
        for task_id in self.best_reward_info.keys():
            task_key = str(task_id)

            bids = self.best_reward_info[task_id].get("bids", [])

            # Count satellites that *currently* think they are the lowest (ties allowed).
            # If you already track this centrally, use that; otherwise compute from bids:
            thinkers_count = self.bidding_info[str(task_id)]["thinks_winner"][self.k]
            print_timestep_summary(self.k, task_id, bids, thinkers_count)

            bri = self.best_reward_info[task_key]
            bi = self.bidding_info[task_key]

            # allow compatibility if bidding_logic still uses "lowest_*" names
            winner_id = bri.get("winner_id") or bri.get("lowest_id")
            winner_rf = bri.get("winner_RF") or bri.get("lowest_RF")
            runner_rf = bri.get("runner_up_RF") or bri.get("second_lowest_RF")

            winner_bid = bri.get("winner_bid")
            if not math.isfinite(winner_bid):
                winner_bid = bri.get("lowest_bid")

            # winner_rf = bri.get("winner_RF")
            # if not math.isfinite(winner_rf):
            #     winner_rf = bri.get("lowest_RF")
            winner_time = bri.get("winner_time") or bri.get("lowest_time") or bri.get("time")

            runner_id = bri.get("runner_up_id") or bri.get("second_lowest_id")
            runner_bid = bri.get("runner_up_bid")
            if not math.isfinite(runner_bid):
                runner_bid = bri.get("second_lowest_bid")
            # runner_rf = bri.get("runner_up_RF")
            # if not math.isfinite(runner_rf):
            #     runner_rf = bri.get("second_lowest_RF")

            # Compute the 2nd price as the winner *thinks* it is:
            winner_perceived_2nd = float("inf")
            if winner_id:
                try:
                    winner_sat = next(s for s in self.satellites if s.name == winner_id)
                    tdict = winner_sat.tasks.get(task_key)
                    if tdict:
                        winner_perceived_2nd = tdict["global"].get("second_lowest_bid", float("inf"))
                except StopIteration:
                    pass

            # If the winner doesn't have a finite second-lowest yet, fall back to the network's runner-up:
            price_2nd = winner_perceived_2nd if \
                        (isinstance(winner_perceived_2nd, (int, float)) and
                        math.isfinite(winner_perceived_2nd)) else \
                        "inf"


            # ensure lists are sized up to index k and assign in-place
            _ensure_ts(bi, "winner_id", self.k, fill=None)[self.k] = winner_id
            _ensure_ts(bi, "winner_bid", self.k, fill=float("inf"))[self.k] = winner_bid
            _ensure_ts(bi, "winner_RF", self.k, fill=0)[self.k] = winner_rf
            _ensure_ts(bi, "RF_time", self.k, fill=0)[self.k] = winner_time

            _ensure_ts(bi, "runner_up_id", self.k, fill=None)[self.k] = runner_id
            _ensure_ts(bi, "runner_up_bid", self.k, fill=float("inf"))[self.k] = runner_bid
            _ensure_ts(bi, "runner_up_RF", self.k, fill=0)[self.k] = runner_rf

            _ensure_ts(bi, "price_paid_2nd", self.k, fill=float("inf"))[self.k] = price_2nd

            # auction (time series)
            # bi["winner_id"].append(bri.get("lowest_id"))
            # bi["winner_bid"].append(bri.get("lowest_bid"))
            # bi["runner_up_id"].append(bri.get("second_lowest_id"))
            # bi["runner_up_bid"].append(bri.get("second_lowest_bid"))
            # bi["price_paid_2nd"].append(bri.get("second_lowest_bid"))

        # ---------------------------------------------------------------------

    def check_break(self):
        """
        Checks if the simulation should break based on the current state of the satellites.
        Can technically be called at any point, and will simply check if the simulation is in such a state that it should end.
        Sets the breaker flag to True if all satellites know all tasks and agree on the best RF for each task.
        :return: None
        """
        # for task_id in self.best_reward_info.keys():
        #     task_key = str(task_id)
        #     bri = self.best_reward_info[task_key]
        #### Check breaking conditions:
        #### Check breaking conditions:
        # --- NEW "STAGNATION BREAK": if no new satellites learn for W steps, stop as UNSOLVED... prevents insanely long runs
        W = 500  # timesteps (dt = 1 s → 500 seconds)
        if self.k >= W:
            # Sum knowledge across all tasks at current step and W steps ago
            total_now = 0
            total_then = 0
            for _tid in range(self.num_tasks):
                _key = str(_tid)
                # knowledge is tracked per-timestep; at this point step() has written index self.k
                # Guard against any missing data by treating absent as 0
                try:
                    total_now += int(self.bidding_info[_key]["knowledge"][self.k])
                except Exception:
                    pass
                try:
                    total_then += int(self.bidding_info[_key]["knowledge"][self.k - W])
                except Exception:
                    pass

            # If knowledge hasn’t increased at all over the last W steps → stagnation
            if total_now == total_then:
                if not hasattr(self, "unsolved_tasks"):
                    self.unsolved_tasks = set()
                self.unsolved_tasks.add("STAGNATION")
                self.breaker = True
                self.end_time = time.time()
                print(
                    f"BREAKER: Stagnation — no knowledge growth for {W} steps (t={self.k - W} → t={self.k}). Marking UNSOLVED.")
                return

        task_knowledge_completion = 0
        RF_knowledge_completion = 0

        for task_id in range(self.num_tasks):
            task_id = str(task_id)
            bri = self.best_reward_info[task_id]

            # network truth: second-best bid
            second_best_net = bri.get("runner_up_bid")
            if not (isinstance(second_best_net, (int, float)) and math.isfinite(second_best_net)):
                second_best_net = bri.get("second_lowest_bid")

            # winner id (from network snapshot)
            winner_id = bri.get("winner_id") or bri.get("lowest_id")

            # winner_sat = next(s for s in self.satellites if s.name == winner_id)
            # second_best_winner = winner_sat.global_knowledge[str(task_id)].get("second_lowest_bid")

            # A) everyone knows the task?
            if self.bidding_info[task_id]['knowledge'][self.k-1] == self.num_satellites:
                print(f"All satellites know task {task_id} at time step {self.k-1}.")
                task_knowledge_completion += 1

                # if fewer than 2 finite bids exist, mark UNSOLVED and stop ===
                bids_list = bri.get("bids", [])
                if len(bids_list) < 2:
                    self.unsolved_tasks.add(task_id)
                    self.breaker = True
                    self.end_time = time.time()
                    print(f"BREAKER: Task {task_id} UNSOLVED — all satellites know it, but <2 valid bids.")
                    return

                # B) exactly one thinker AND the winner knows the correct second-best price
                if self.bidding_info[task_id]['thinks_winner'][self.k-1] == 1:
                    winner_knows_price = False
                    if winner_id and isinstance(second_best_net, (int, float)) and math.isfinite(second_best_net):
                        # fetch winner sat and read its own global view
                        try:
                            winner_sat = next(s for s in self.satellites if s.name == winner_id)
                            second_best_winner = winner_sat.tasks[task_id]['global'].get('second_lowest_bid',
                                                                                          float('inf'))
                            winner_knows_price = (
                                    isinstance(second_best_winner, (int, float)) and
                                    math.isfinite(second_best_winner) and
                                    abs(second_best_winner - second_best_net) <= 1e-9
                            )
                        except StopIteration:
                            winner_knows_price = False
                    if winner_knows_price:
                        RF_knowledge_completion += 1
                        print("All satellites agree on a winner AND the winner knows the correct 2nd price.")
                    else:
                        print("A single winner is agreed, but the winner does NOT yet know the correct 2nd price.")
                    # RF_knowledge_completion += 1
                    # print("All satellites agree that one satellite is the winner.")

        if task_knowledge_completion == self.num_tasks and RF_knowledge_completion == self.num_tasks:
            self.breaker = True
            self.end_time = time.time()

            print("BREAKER: All satellites know all tasks and agree on the best RF for each task. When you advance to the next step, the sim will end.")
        
        if self.k >= self.num_steps:
            self.breaker = True
            self.end_time = time.time()
            print(f"BREAKER: Simulation ended at time step {self.k} after {self.end_time - self.start_time:.2f} seconds. Reached maximum number of steps.")

    def save_data(self, outputfolder=None):
        """
        Saves the simulation data to an Excel file.
        The file name encodes REAL simulation parameters (post-preset) and is placed under:
            <outputfolder>/<DDMMYYYY>/<run_name>_<YYYYMMDD_HHMMSS>/
        """
        if not outputfolder:
            return  # explicitly do nothing if no output folder was provided

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        date_folder = now.strftime("%d%m%Y")  # e.g., 22092025

        # --- Ground-truth counts (REAL simulation numbers) ---
        real_nsat = len(getattr(self, "satellites", [])) or int(getattr(self, "num_satellites", 0))
        real_ncn = sum(1 for s in getattr(self, "satellites", []) if getattr(s, "flags", {}).get("central", 0) == 1)
        if real_ncn == 0:
            # fallback to planned number if for some reason flags aren't set
            real_ncn = int(getattr(self, "num_central_nodes", 0))

        # Approximate number of OMNI CNs from the configured fraction
        try:
            real_nomni = int(round(float(self.omni_fraction_CNs) * real_ncn))
        except Exception:
            real_nomni = 0

        # --- Build folder structure:
        # <outputfolder>/<DDMMYYYY>/<S{ns}_CN{ncn}_OMNICN{nomni}_C{constellation}> ---
        const_str = f"{self.constellation}"
        cfg_folder = f"S{real_nsat}_CN{real_ncn}_OMNICN{real_nomni}_C{const_str}"
        root_folder = os.path.join(outputfolder, date_folder, cfg_folder)
        os.makedirs(root_folder, exist_ok=True)
        print(f"Created output folder: {root_folder}")

        # --- Build filename with the convention ---
        # RFhist_T{tasks}_S{nsat}_CN{ncn}_OCN{ocnfrac}_k{kvar}_C{constellation}_TD{0|1}_Rand{0|1}_Seed{seed}_{timestamp}.xlsx
        tasks_str = f"{self.num_tasks}"
        nsat_str = f"{real_nsat}"
        ncn_str = f"{real_ncn}"
        ocnfrac_str = f"{float(self.omni_fraction_CNs):.2f}"
        kvar_str = f"{float(self.k_var):g}"
        const_str = f"{self.constellation}"
        td_flag = "1" if bool(self.time_dependency) else "0"
        rand_flag = "1" if bool(self.randomize_constellation) else "0"
        seed_val = getattr(self, "seed", None)
        seed_str = str(seed_val) if seed_val is not None else "NA"

        status_tag = "UNSOLVED__" if getattr(self, "unsolved_tasks", set()) else ""

        filename = os.path.join(
            root_folder,
            f"{status_tag}T{tasks_str}_S{nsat_str}_CN{ncn_str}_OCN{ocnfrac_str}"
            f"_k{kvar_str}_C{const_str}_TD{td_flag}_Rand{rand_flag}_Seed{seed_str}_{timestamp}.xlsx"
        )

        # --- Collect sheets ---
        sheets = {}
        for task_key, data in self.bidding_info.items():
            n = len(data.get("timestep", []))
            if n == 0:
                sheets[f"task_{task_key}"] = pd.DataFrame({"timestep": []})
                continue

            df = pd.DataFrame({
                "time_step": _list_or_default(data, "timestep", n),
                "RF_time": _list_or_default(data, "RF_time", n),
                "knowledge": _list_or_default(data, "knowledge", n, 0),
                "thinks_winner": _list_or_default(data, "thinks_winner", n, 0),

                "winner_id": _list_or_default(data, "winner_id", n),
                "winner_bid": _list_or_default(data, "winner_bid", n),
                "winner_RF": _list_or_default(data, "winner_RF", n),

                "runner_up_id": _list_or_default(data, "runner_up_id", n),
                "runner_up_bid": _list_or_default(data, "runner_up_bid", n),
                "runner_up_RF": _list_or_default(data, "runner_up_RF", n),

                "price_paid_2nd": _list_or_default(data, "price_paid_2nd", n),
            })
            sheets[f"task_{task_key}"] = df

        if not sheets:
            sheets["summary"] = pd.DataFrame({"info": ["No data collected"]})

        # --- Write workbook ---
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        print(f"Simulation data saved to {filename}")

        return filename

    def get_state(self):
        """
        Return a full deepcopy of the current simulation object.
        This captures all internal state (including satellites, flags, subsystems, etc.).
        """

        return copy.deepcopy(self)
    
    def load_state(self, snapshot):
        """
        Overwrite the current simulation object's internal state
        with that from a saved snapshot (also a Sim instance).
        This does NOT reassign `self`, but copies all attributes in-place.

        """
        for attr in vars(snapshot):
            setattr(self, attr, copy.deepcopy(getattr(snapshot, attr)))
    
    def load_step(self, timestep:int, history):
        """
        Load the simulation state from a previous timestep.
        :param timestep: The index of the timestep to load from the history.
        :return: None
        """
        if timestep < 0 or timestep >= len(history):
            raise IndexError("Invalid timestep index.")
        self.load_state(history[timestep])

    def run_full(self, outputfolder=None, savedata=False, savehistory=False):
        """
        Self-contained runnner for running the full simulation, from start to end.
        :param outputfolder: Folder where the data will be saved. If None, it will not save the data.
        :param savedata: If True, saves the simulation data to an Excel file.
        :param savehistory: If True, saves the history of the simulation steps to a file.
        :return: None
        """
        history = []
        if not outputfolder and savedata:
            raise ValueError("Please specify an output folder if you want to save the data.")
        if outputfolder and not savedata:
            print("Warning: You specified an output folder but not to save the data. No data will be saved.")

        # Run the first step to initialize tasks in one satellite
        self.first_step()

        # Run the simulation for each time step
        while True:
            if self.breaker:
                print(f"Simulation ended at time step {self.k-1} after {self.end_time - self.start_time:.2f} seconds.")
                break
            self.step()
            if savehistory:
                history.append(self.get_state())
            self.check_break()
            self.k += 1  # Increment the current time step

        saved_path = None
        # Save the data if required
        if savedata:
            saved_path = self.save_data(outputfolder)

        if savehistory:
            history_filename = rf'presaved_sims\history_{self.name}.pkl'
            with open(history_filename, 'wb') as f:
                pickle.dump(history, f)
            print(f"Simulation history saved to {history_filename}")

            # --- return a compact summary for parallel runners ---
        elapsed = (getattr(self, "end_time", time.time()) - self.start_time)
        return {
            "run_name": self.name,
            "elapsed_s": elapsed,
            "timesteps": self.k - 1,
            "saved_file": saved_path,
        }



# ============================================================================
# 3D VISUALIZATION - Toggle this to visualize the constellation
# ============================================================================
ENABLE_3D_VISUALIZATION = True  # Set to True to enable interactive 3D visualization

# Visualization is now in a separate module to avoid interference
def visualize_constellation_3d_wrapper(sim, max_timesteps=100, update_interval=50,
                                      sample_interval=None, max_sim_time=None, save_path=None):
    """
    Wrapper that calls the standalone visualization module.
    This keeps visualization code completely separate from simulation.py

    Args:
        sim: Simulation object
        max_timesteps: Max timesteps if using sequential mode
        update_interval: Milliseconds between animation frames
        sample_interval: Sample every N seconds (e.g., 100)
        max_sim_time: Total simulation time in seconds (e.g., 10000)
        save_path: Where to save HTML file (None=temp, "dir/"=auto-name, "file.html"=specific)
    """
    try:
        from visualize_3d_constellation import visualize_constellation_3d
        return visualize_constellation_3d(sim, max_timesteps=max_timesteps,
                                         update_interval=update_interval, silent=True,
                                         sample_interval=sample_interval,
                                         max_sim_time=max_sim_time,
                                         save_path=save_path)
    except ImportError as e:
        print(f"ERROR: Could not import visualization module: {e}")
        print("Make sure visualize_3d_constellation.py is in the same directory.")


if __name__ == "__main__":
    sim = Sim(run_name='test',
              verbosity=False,
              type='partially_centralized',
              num_tasks=1,
              num_satellites=1000,
              fraction_CNs=0.2,
              # omni_fraction_CNs = 0.0 -> legacy behavior (single initial observer)
              # omni_fraction_CNs = 1.0 -> all CNs omniscent at t=0
              omni_fraction_CNs=0.0,
              k_var=10.0,
              # Choose from one of the constellation presets:
              # "walker_delta_10x50"   -> Generic Walker-Delta, 10 planes × 50 sats, i≈53°, h≈550 km
              # "walker_delta_20x30"   -> Generic Walker-Delta, 20 planes × 30 sats, i≈53°, h≈550 km
              # "walker_star_20x30"    -> Walker-Star variant, 20 planes × 30 sats, phasing F=7
              # "starlink_shell_1"     -> Approx Starlink shell (72 planes × 22 sats, i≈53°, h≈550 km)
              # "planet_sso"           -> Approx Planet SuperDove cluster (12 planes × 12 sats, i≈98°, h≈475 km)
              constellation='starlink_shell_1',
              # Ignores num_satellites set above and use preset if True
              # Ignores preset and uses num_satellites if False
              preset_size=False,
              time_dependency=True,
              randomize_constellation=True,
              # for reproducibility:
              seed=8
              )

    # Run 3D visualization if enabled
    if ENABLE_3D_VISUALIZATION:
        # Sample every 10 seconds for 10000 seconds total (1001 frames - smoother motion!)
        # Save to visualizations/ folder with auto-generated filename
        viz_path = visualize_constellation_3d_wrapper(
            sim,
            sample_interval=5,     # Every 100 seconds for testing
            max_sim_time=5000,       # Quick test: 1000 seconds
            update_interval=100,     # screen refreshing time in ms. 100(ms) is 10FPS
            save_path="visualizations/"  # Auto-generates filename with timestamp
        )
        print(f"\n[SAVED] Visualization saved permanently to: {viz_path}")
        print(f"[INFO] You can reopen this file anytime - no need to regenerate!")
    else:
        sim.run_full(outputfolder=r'C:\Users\bocca\fssv2', savedata=True, savehistory=False)








