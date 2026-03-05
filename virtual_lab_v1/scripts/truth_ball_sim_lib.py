"""
truth_ball_sim_lib.py - Importable wrapper for 14_truth_ball_sim.py
===================================================================
Python cannot import modules starting with digits, so this module
re-exports the core functions via importlib.
"""
import importlib.util
from pathlib import Path

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "truth_ball_sim", _here / "14_truth_ball_sim.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export public API
simulate_sphere = _mod.simulate_sphere
compute_event_rate = _mod.compute_event_rate
compute_derived_params = _mod.compute_derived_params
quick_sim = _mod.quick_sim
compute_drag_force = _mod.compute_drag_force
compute_du_dt = _mod.compute_du_dt
compute_pin_statistics = _mod.compute_pin_statistics
