import uuid
from typing import Any, List, Optional

from openenv.core import Environment

from warehouse_robot_nav.models import (
    WarehouseRobotAction,
    WarehouseRobotObservation,
    WarehouseRobotState,
)

DIRECTION_MAP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

TASKS = {
    "easy": {
        "grid_size": 5,
        "obstacles": [[1, 2], [3, 1]],
        "robot_start": [4, 0],
        "goal": [0, 4],
        "max_steps": 20,
        "desc": "Simple 5x5 warehouse with 2 obstacles",
    },
    "medium": {
        "grid_size": 7,
        "obstacles": [[1, 1], [1, 5], [2, 3], [3, 5], [4, 1], [5, 4]],
        "robot_start": [6, 0],
        "goal": [0, 6],
        "max_steps": 35,
        "desc": "7x7 warehouse with 6 obstacles",
    },
    "hard": {
        "grid_size": 10,
        "obstacles": [
            [1, 1], [1, 4], [2, 6], [3, 1], [3, 8],
            [4, 3], [5, 0], [5, 5], [6, 2], [6, 7],
            [7, 4], [8, 1], [8, 6],
        ],
        "robot_start": [9, 0],
        "goal": [0, 9],
        "max_steps": 60,
        "desc": "10x10 warehouse with 13 obstacles",
    },
}


class WarehouseRobotEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.current_task = "easy"
        self.pending_task: Optional[str] = None
        self.grid_size = 5
        self.robot_pos = [4, 0]
        self.goal_pos = [0, 4]
        self.obstacles: List[List[int]] = []
        self._step_count = 0
        self.max_steps = 20
        self.collisions = 0
        self._episode_id = ""
        self.initial_distance = 0

    def set_task(self, task_name: str):
        if task_name in TASKS:
            self.pending_task = task_name

    def _dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _render_grid(self) -> str:
        lines = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if [r, c] == self.robot_pos:
                    row.append("R")
                elif [r, c] == self.goal_pos:
                    row.append("G")
                elif [r, c] in self.obstacles:
                    row.append("X")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)

    def _compute_score(self) -> float:
        dist = self._dist(self.robot_pos, self.goal_pos)
        goal_reached = dist == 0
        progress = max(0.0, 1.0 - dist / max(self.initial_distance, 1)) * 0.5
        goal_bonus = 0.3 if goal_reached else 0.0
        efficiency = (
            max(0.0, 1.0 - self._step_count / max(self.max_steps, 1)) * 0.2
            if goal_reached
            else 0.0
        )
        return min(1.0, progress + goal_bonus + efficiency)

    def _obs(self, message: str, reward: float, done: bool) -> WarehouseRobotObservation:
        return WarehouseRobotObservation(
            grid=self._render_grid(),
            robot_position=list(self.robot_pos),
            goal_position=list(self.goal_pos),
            message=message,
            manhattan_distance=self._dist(self.robot_pos, self.goal_pos),
            step_count=self._step_count,
            max_steps=self.max_steps,
            task_name=self.current_task,
            score=self._compute_score(),
            reward=reward,
            done=done,
        )

    def reset(self, seed=None, episode_id=None, **kwargs) -> WarehouseRobotObservation:
        # Check global class-level pending task first
        global_task = getattr(WarehouseRobotEnvironment, "_global_pending_task", None)
        if global_task:
            self.current_task = global_task
            WarehouseRobotEnvironment._global_pending_task = None

        if self.pending_task:
            self.current_task = self.pending_task
            self.pending_task = None
        task = TASKS[self.current_task]
        self.grid_size = task["grid_size"]
        self.robot_pos = list(task["robot_start"])
        self.goal_pos = list(task["goal"])
        self.obstacles = [list(o) for o in task["obstacles"]]
        self.max_steps = task["max_steps"]
        self._step_count = 0
        self.collisions = 0
        self._episode_id = episode_id or str(uuid.uuid4())
        self.initial_distance = self._dist(self.robot_pos, self.goal_pos)
        msg = (
            f"Task '{self.current_task}': {task['desc']}. "
            f"Navigate R to G, avoid X. Directions: up, down, left, right."
        )
        return self._obs(msg, reward=0.0, done=False)

    def step(self, action: WarehouseRobotAction, **kwargs) -> WarehouseRobotObservation:
        direction = action.direction.lower().strip()
        self._step_count += 1

        # Invalid direction
        if direction not in DIRECTION_MAP:
            done = self._step_count >= self.max_steps
            msg = f"Invalid direction '{direction}'. Use: up/down/left/right."
            if done:
                msg += f" Max steps. Score: {self._compute_score():.2f}"
            return self._obs(msg, reward=-0.1, done=done)

        dr, dc = DIRECTION_MAP[direction]
        nr, nc = self.robot_pos[0] + dr, self.robot_pos[1] + dc

        # Wall collision
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            self.collisions += 1
            done = self._step_count >= self.max_steps
            msg = f"Wall collision moving {direction}! Stay in bounds."
            if done:
                msg += f" Max steps. Score: {self._compute_score():.2f}"
            return self._obs(msg, reward=-0.2, done=done)

        # Obstacle collision
        if [nr, nc] in self.obstacles:
            self.collisions += 1
            done = self._step_count >= self.max_steps
            msg = f"Obstacle at ({nr},{nc})! Choose another direction."
            if done:
                msg += f" Max steps. Score: {self._compute_score():.2f}"
            return self._obs(msg, reward=-0.15, done=done)

        # Valid move
        old_dist = self._dist(self.robot_pos, self.goal_pos)
        self.robot_pos = [nr, nc]
        new_dist = self._dist(self.robot_pos, self.goal_pos)

        # Goal reached
        if new_dist == 0:
            score = self._compute_score()
            return self._obs(f"Goal reached! Score: {score:.2f}", reward=1.0, done=True)

        # Progress reward
        reward = 0.1 if new_dist < old_dist else -0.05
        done = self._step_count >= self.max_steps
        msg = f"Moved {direction} to ({nr},{nc}). Distance: {new_dist}."
        if done:
            msg += f" Max steps. Score: {self._compute_score():.2f}"
        return self._obs(msg, reward=reward, done=done)

    def state(self) -> WarehouseRobotState:
        return WarehouseRobotState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self.current_task,
            collisions=self.collisions,
        )
