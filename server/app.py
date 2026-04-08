from openenv.core import create_app
from warehouse_robot_nav.models import WarehouseRobotAction, WarehouseRobotObservation
from warehouse_robot_nav.environment import WarehouseRobotEnvironment

app = create_app(
    env=WarehouseRobotEnvironment,
    action_cls=WarehouseRobotAction,
    observation_cls=WarehouseRobotObservation,
    env_name="warehouse_robot_nav",
)


# Custom endpoint to switch tasks between episodes
from fastapi import FastAPI
from pydantic import BaseModel


class SetTaskRequest(BaseModel):
    task_name: str


@app.post("/set_task")
async def set_task(req: SetTaskRequest):
    # This sets a global pending task for the next reset
    WarehouseRobotEnvironment._global_pending_task = req.task_name
    return {"status": "ok", "task": req.task_name}


@app.get("/tasks")
async def list_tasks():
    from warehouse_robot_nav.environment import TASKS
    return {"tasks": list(TASKS.keys())}
