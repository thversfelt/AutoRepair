import random
from typing import Dict, List


def tarantula_suspiciousness(
        statement: int,
        passed: Dict[int, int], failed: Dict[int, int],
        total_passed: int, total_failed: int) -> float:
    percentage_failed = failed[statement] / total_failed if total_failed > 0 else 0  # Prevent division by 0 error.
    percentage_passed = passed[statement] / total_passed if total_passed > 0 else 1  # Prevent division by 0 error.
    return percentage_failed / (percentage_passed + percentage_failed)


def roulette_wheel_selection(population: Dict[int, float]) -> int:
    total_value = sum(population.values())
    selection_value = random.uniform(0, total_value)
    current_value = 0
    for key, value in population.items():
        current_value += value
        if current_value >= selection_value:
            return key


def dominates(one: List[float], other: List[float]) -> bool:
    for i in range(len(one)):
        if other[i] > one[i]:
            return False  # One is dominated by at least one objective of the other

    for i in range(len(one)):
        if one[i] > other[i]:
            return True  # The other does not dominate one, and one dominates at least one objectives of the other.

    return False  # The other does not dominate one, but one does not dominate any objective of the other.
