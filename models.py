from typing import List, Optional
from openenv.core import Action, Observation, State


class WarehouseRobotAction(Action):
    direction: str  # "up", "down", "left", "right"


class WarehouseRobotObservation(Observation):
    grid: str = ""
    robot_position: List[int] = [0, 0]
    goal_position: List[int] = [0, 0]
    message: str = ""
    manhattan_distance: int = 0
    step_count: int = 0
    max_steps: int = 0
    task_name: str = "easy"
    score: float = 0.0


class WarehouseRobotState(State):
    task_name: str = "easy"
    collisions: int = 0
