"""
Optimization Module for Return-Load Fleet Allocation.
Provides MIP solver wrappers using Google OR-Tools.
"""
from typing import Dict, Any, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from matching import ReturnLoadMatcher

class FleetReturnOptimizer:
    """
    Fleet-wide return load MIP optimization wrapper using Google OR-Tools.
    """
    def __init__(self, data_dir: Any = None):
        self.matcher = ReturnLoadMatcher(data_dir)

    def optimize_assignments(self, high_risk_vehicles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs OR-Tools MIP solver to assign optimal return loads to high-risk vehicles."""
        return self.matcher.optimize_fleet_assignments(high_risk_vehicles)
