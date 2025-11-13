# run_parallel.py — IDE-friendly (PyCharm) + CLI compatible
import time
import multiprocessing as mp
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # harmless on non-Apple
import sys
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from itertools import product
import argparse
from typing import Any, Dict, List
from types import SimpleNamespace

from simulation import Sim  # your live simulator


# ===================== USER GRID (EDIT THESE LISTS) =========================
NUM_SATELLITES    = [500, 1000, 2000, 3000]             # iterate this (x4)
FRACTION_CNS      = [0.10, 0.25, 0.50]                  # iterate this (x3)
K_VAR_LIST        = [5.0]                     # iterate this (x3)
FRACTION_OMNI_CNS = [0.0, 0.25, 0.50, 1.0]  # iterate this (x6)
CONSTELLATIONS    = [
    "walker_delta_10x50",
    "walker_delta_20x30",
    "starlink_shell_1",
    "planet_sso",
]



# ===================== EXECUTION CONTROLS ===================================
# Default worker count used if neither CLI nor IDE config sets it:
PROCESSES         = 10
BASE_RUN_NAME     = "grid"                               # prefix for run_name
OUT_DIR           = os.path.dirname(os.path.abspath(__file__))
SWEEP_SEEDS       = True                                 # seed = base + idx (reproducible)
BASE_SEED         = 0
SAVE_DATA         = True
SAVE_HISTORY      = False

# IMPORTANT: if you vary num_satellites, your preset can override it.
# Set this to False so the explicit num_satellites takes effect.
# If you want the preset's built-in size instead, set True.
PRESET_SIZE       = False
# ============================================================================

# ===================== IDE RUN CONFIG (PyCharm) =============================
# Set USE_CLI=False to ignore command-line flags and run with the variables below.
USE_CLI = True

IDE = SimpleNamespace(
    # Choose "grid" or "montecarlo"
    mode="montecarlo",

    # Parallelism and repetitions
    processes=PROCESSES,
    repeats=60,

    # --- Fixed values (used for grid tagging and for montecarlo when not swept)
    # If ns is None, we'll use preset size (unless overridden by policy below)
    ns=None,                       # e.g., 1000 or None
    frac=0.20,                     # fraction_CNs
    omni=0.0,                      # omni_fraction_CNs
    const="starlink_shell_1",      # one from CONSTELLATIONS
    kvar=2.0,                      # k_var when not sweeping k_var

    # --- Monte Carlo mode (only if mode="montecarlo")
    # Sweep ONE of: "ns", "frac", "omni", "k_var", "const"
    sweep="k_var",

    # --- Preset-size policy for IDE:
    #   "auto" -> if sweep=="ns": False; else True when ns is None, otherwise False
    #   True   -> always use preset sizes
    #   False  -> never use preset sizes (explicit ns always applies)
    preset_size_policy="auto",
)
# ============================================================================


@contextmanager
def _silent_stdio():
    """Redirect stdout/stderr to devnull so child sims don't spam and eat RAM."""
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        yield


def _run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Worker: instantiate Sim(**cfg) and run."""
    out_dir = cfg.pop("out_dir", None)
    savedata = cfg.pop("savedata", True)
    savehistory = cfg.pop("savehistory", False)

    with _silent_stdio():
        sim = Sim(**cfg)  # type: ignore
        t0 = time.time()
        summary = sim.run_full(outputfolder=out_dir, savedata=savedata, savehistory=savehistory)
        t1 = time.time()

    return {
        "run_name": summary.get("run_name", getattr(sim, "name", "")),
        "timesteps": summary.get("timesteps", getattr(sim, "k", 0)),
        "elapsed_s": summary.get("elapsed_s", t1 - t0),
        "saved_file": summary.get("saved_file"),
        "seed": getattr(sim, "seed", None),
        "num_satellites": getattr(sim, "num_satellites", None),
        "fraction_CNs": getattr(sim, "fraction_CNs", None),
        "k_var": getattr(sim, "k_var", None),
        "constellation": getattr(sim, "constellation", None),
    }


def _mk_cfg(idx: int, ns: int, frac: float, frac_omni: float, kv: float, constel: str,
            preset_flag: bool) -> Dict[str, Any]:
    seed = BASE_SEED + idx if SWEEP_SEEDS else None
    run_name = f"{BASE_RUN_NAME}_S{ns}_CN{int(round(frac*100))}_k{kv:g}_{constel}_r{idx:03d}"
    return dict(
        # Sim(...) arguments
        run_name=run_name,
        verbosity=False,
        type="partially_centralized",
        num_tasks=1,
        num_satellites=ns,
        fraction_CNs=frac,
        dt=1,
        duration=6 * 3600,
        k_var=kv,
        constellation=constel,
        preset_size=preset_flag,
        time_dependency=True,
        omni_fraction_CNs=frac_omni,
        seed=seed,
        randomize_constellation=True,
        # run_full() options (consumed in _run_one)
        out_dir=OUT_DIR,
        savedata=SAVE_DATA,
        savehistory=SAVE_HISTORY,
    )


def run_parallel(configs: List[Dict[str, Any]], processes: int) -> List[Dict[str, Any]]:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    total = len(configs)
    if total == 0:
        print("Nothing to run.")
        return []

    results: List[Dict[str, Any]] = []
    start = time.time()
    done = 0

    with mp.Pool(processes=processes, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(_run_one, configs, chunksize=1):
            results.append(res)
            done += 1
            avg = (time.time() - start) / max(done, 1)
            eta_s = (total - done) * avg
            pct = 100.0 * done / total
            sys.stdout.write(
                f"\rProgress: {done}/{total} ({pct:5.1f}%) | "
                f"avg {avg:6.2f}s/run | ETA {eta_s / 60:5.1f} min"
            )
            sys.stdout.flush()

    print()  # newline after finishing the bar
    return results


def _build_args_from_cli():
    p = argparse.ArgumentParser(description="Run FSS simulations in parallel.")
    p.add_argument("--mode", choices=["grid", "montecarlo"], default="grid")
    p.add_argument("--processes", type=int, default=None)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--ns", type=int, default=None)
    p.add_argument("--frac", type=float, default=FRACTION_CNS[0])
    p.add_argument("--omni", type=float, default=FRACTION_OMNI_CNS[0])
    p.add_argument("--const", choices=CONSTELLATIONS, default=CONSTELLATIONS[0])
    p.add_argument(
        "--sweep",
        choices=["ns", "frac", "omni", "k_var", "const"],
        default=None,
        help="Parameter to iterate when mode=montecarlo (values taken from the lists at the top of the file)."
    )
    p.add_argument(
        "--kvar",
        type=float,
        default=K_VAR_LIST[0],
        help="Fixed k_var value when not sweeping k_var."
    )
    args = p.parse_args()
    # add a field for IDE-like policy (CLI keeps original behavior below)
    args.preset_size_policy = None
    return args


def _build_args_from_ide():
    # Validate a few IDE inputs early
    if IDE.mode not in ("grid", "montecarlo"):
        raise ValueError("IDE.mode must be 'grid' or 'montecarlo'")
    if IDE.mode == "montecarlo" and IDE.sweep not in ("ns", "frac", "omni", "k_var", "const"):
        raise ValueError("IDE.sweep must be one of: ns, frac, omni, k_var, const")

    return SimpleNamespace(
        mode=IDE.mode,
        processes=IDE.processes,
        repeats=IDE.repeats,
        ns=IDE.ns,
        frac=IDE.frac,
        omni=IDE.omni,
        const=IDE.const,
        sweep=(IDE.sweep if IDE.mode == "montecarlo" else None),
        kvar=IDE.kvar,
        preset_size_policy=IDE.preset_size_policy,
    )


if __name__ == "__main__":
    # Choose config source
    args = _build_args_from_cli() if USE_CLI else _build_args_from_ide()
    PROCS = args.processes if (args.processes and args.processes > 0) else PROCESSES

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ---------- Build run configurations -------------------------------------
    cfgs: List[Dict[str, Any]] = []

    if args.mode == "montecarlo":
        # Choose sweep set
        if args.sweep == "ns":
            sweep_vals = NUM_SATELLITES
        elif args.sweep == "frac":
            sweep_vals = FRACTION_CNS
        elif args.sweep == "omni":
            sweep_vals = FRACTION_OMNI_CNS
        elif args.sweep == "k_var":
            sweep_vals = K_VAR_LIST
        elif args.sweep == "const":
            sweep_vals = CONSTELLATIONS
        else:
            raise SystemExit("When mode=montecarlo you must set sweep in IDE or --sweep via CLI.")

        # Determine preset-size policy for IDE; CLI retains legacy behavior
        def _preset_policy(ns_fixed: Any, sweep_key: str) -> bool:
            if args.preset_size_policy is None:
                # CLI path (legacy): if sweeping NS -> False; else True iff ns was omitted
                if sweep_key == "ns":
                    return False
                return (args.ns is None)
            if args.preset_size_policy == "auto":
                if sweep_key == "ns":
                    return False
                return (ns_fixed is None)
            return bool(args.preset_size_policy)

        idx = 0
        for val in sweep_vals:
            for _ in range(args.repeats):
                if args.sweep == "ns":
                    ns_for_sim = int(val)
                    frac = args.frac
                    omni = args.omni
                    kv = args.kvar
                    constel = args.const
                elif args.sweep == "frac":
                    ns_for_sim = args.ns if args.ns is not None else 0
                    frac = float(val)
                    omni = args.omni
                    kv = args.kvar
                    constel = args.const
                elif args.sweep == "omni":
                    ns_for_sim = args.ns if args.ns is not None else 0
                    frac = args.frac
                    omni = float(val)
                    kv = args.kvar
                    constel = args.const
                elif args.sweep == "k_var":
                    ns_for_sim = args.ns if args.ns is not None else 0
                    frac = args.frac
                    omni = args.omni
                    kv = float(val)
                    constel = args.const
                else:  # const
                    ns_for_sim = args.ns if args.ns is not None else 0
                    frac = args.frac
                    omni = args.omni
                    kv = args.kvar
                    constel = str(val)

                preset_flag = _preset_policy(args.ns, args.sweep)
                run_name = f"{BASE_RUN_NAME}_MC_{args.sweep}_{idx:03d}"
                cfg = dict(
                    run_name=run_name,
                    verbosity=False,
                    type="partially_centralized",
                    num_tasks=1,
                    num_satellites=ns_for_sim,
                    fraction_CNs=frac,
                    dt=1,
                    duration=6 * 3600,
                    k_var=kv,
                    constellation=constel,
                    preset_size=preset_flag,
                    time_dependency=True,
                    omni_fraction_CNs=omni,
                    seed=BASE_SEED + idx if SWEEP_SEEDS else None,
                    randomize_constellation=True,
                    out_dir=OUT_DIR,
                    savedata=SAVE_DATA,
                    savehistory=SAVE_HISTORY,
                )
                cfgs.append(cfg)
                idx += 1

        print(f"[MC] {len(cfgs)} runs sweeping '{args.sweep}' over {len(sweep_vals)} values | "
              f"repeats={args.repeats} | processes={PROCS}")

    else:  # grid
        combos = list(product(
            NUM_SATELLITES,        # ns
            FRACTION_CNS,          # frac
            FRACTION_OMNI_CNS,     # omni
            K_VAR_LIST,            # k_var
            CONSTELLATIONS         # const
        ))
        idx = 0
        for _ in range(args.repeats):
            for (ns, frac, frac_omni, kv, constel) in combos:
                cfgs.append(_mk_cfg(idx, ns, frac, frac_omni, kv, constel, PRESET_SIZE))
                idx += 1

    # --- Startup summary ------------------------------------------------------
    print("== run_parallel started ==")
    print(f"Mode: {args.mode} | Processes: {PROCS} | Repeats: {args.repeats}")
    print(f"Output dir: {OUT_DIR}")

    if args.mode == "grid":
        print("Grid sizes:",
              f"NS={len(NUM_SATELLITES)}, FRAC_CNS={len(FRACTION_CNS)}, "
              f"OMNI={len(FRACTION_OMNI_CNS)}, K={len(K_VAR_LIST)}, "
              f"CONST={len(CONSTELLATIONS)}")
        print(f"preset_size={PRESET_SIZE}")
    else:
        fixed_ns = 'preset' if (args.sweep != 'ns' and
                                ((args.preset_size_policy in (None, 'auto') and args.ns is None) or
                                 args.preset_size_policy is True)) else (args.ns if args.sweep != 'ns' else 'swept')
        print("MC setup:",
              f"Sweep={args.sweep} | Fixed -> NS={fixed_ns}, "
              f"FRAC_CNS={'swept' if args.sweep=='frac' else args.frac}, "
              f"OMNI={'swept' if args.sweep=='omni' else args.omni}, "
              f"K={'swept' if args.sweep=='k_var' else args.kvar}, "
              f"CONST={'swept' if args.sweep=='const' else args.const}, "
              f"preset_size={'False' if args.sweep=='ns' else ('auto' if (args.preset_size_policy in (None,'auto')) else args.preset_size_policy)}")

    t0 = time.time()
    results = run_parallel(cfgs, PROCS)

    if results:
        by_const = {}
        for r in results:
            c = r.get("constellation")
            n = r.get("num_satellites")
            if c and n is not None:
                by_const[c] = n
        eff_summary = ", ".join(f"{c}→{n}" for c, n in by_const.items())
        print(f"\nEffective satellite counts (from presets): {eff_summary or 'n/a'}")

    t1 = time.time()
    print(f"Done. Completed {len(results)} runs in {t1 - t0:.2f}s")
