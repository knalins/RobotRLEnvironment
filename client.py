from openenv.core.env_client import EnvClient
from models import WarehouseRobotAction, WarehouseRobotObservation, WarehouseRobotState


class WarehouseRobotEnv(EnvClient[WarehouseRobotAction, WarehouseRobotObservation, WarehouseRobotState]):
    Action = WarehouseRobotAction
    Observation = WarehouseRobotObservation
    State = WarehouseRobotState
