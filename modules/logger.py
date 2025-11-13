# modules/auction_logging.py
from typing import List, Dict, Optional, Tuple
import math

Bid = Dict[str, float]  # {"id","bid","RF","time"} in your repo

def _ensure_ts(bi, key, k, fill=0):
    # bi: self.bidding_info[task_id]
    if key not in bi or not isinstance(bi[key], list):
        bi[key] = []
    while len(bi[key]) <= k:
        bi[key].append(fill)
    return bi[key]

def _fmt_num(x, nd=3):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def _fmt_time(x):
    if x is None:
        return "—"
    try:
        return str(int(round(float(x))))
    except Exception:
        return "—"

def _fmt_bid(b: Optional[Bid]) -> str:
    if not b:
        return "—"
    bid = _fmt_num(b.get("bid"))
    rf  = _fmt_num(b.get("RF"))
    t   = _fmt_time(b.get("time"))
    return f"{b.get('id','?')} | bid={bid} | RF={rf} | t={t}"

def best_two_from(bids: List[Bid]) -> Tuple[Optional[Bid], Optional[Bid]]:
    if not bids:
        return None, None
    # keep only entries with numeric bids
    def _is_num(v):
        try:
            v = float(v)
            return not math.isnan(v) and not math.isinf(v)
        except Exception:
            return False
    s = sorted([e for e in bids if _is_num(e.get("bid"))], key=lambda x: float(x["bid"]))
    best = s[0] if len(s) >= 1 else None
    second = s[1] if len(s) >= 2 else None
    return best, second

def print_bid_event(task_id: str, timestep: int, new_bid: Bid, best: Optional[Bid], second: Optional[Bid]) -> None:
    # 1) New candidate appears / updates bid
    print(f"[t={timestep}] CANDIDATE task {task_id}: {_fmt_bid(new_bid)}")

    # 2) Immediately show leaders (lowest & second-lowest)
    print(f"          LEADERS → lowest: {_fmt_bid(best)} | second: {_fmt_bid(second)}")

def print_timestep_summary(timestep: int, task_id: str, bids: List[Bid], thinkers_count: int) -> None:
    best, second = best_two_from(bids)
    print(f"[t={timestep}] SUMMARY task {task_id}: "
          f"{thinkers_count} sat(s) think they’re lowest | "
          f"lowest={_fmt_bid(best)} | second={_fmt_bid(second)}")

def _list_or_default(d, key, n, default=None):
    if key in d and isinstance(d[key], list):
        return d[key]
    return [default] * n