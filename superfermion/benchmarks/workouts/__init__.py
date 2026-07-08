"""Benchpress-style workout definitions."""

from superfermion.benchmarks.workouts.construction import CONSTRUCTION_WORKOUTS
from superfermion.benchmarks.workouts.manipulation import MANIPULATION_WORKOUTS
from superfermion.benchmarks.workouts.transpilation import TRANSPILATION_WORKOUTS

ALL_WORKOUTS = {
    **CONSTRUCTION_WORKOUTS,
    **MANIPULATION_WORKOUTS,
    **TRANSPILATION_WORKOUTS,
}

__all__ = [
    "CONSTRUCTION_WORKOUTS",
    "MANIPULATION_WORKOUTS",
    "TRANSPILATION_WORKOUTS",
    "ALL_WORKOUTS",
]
