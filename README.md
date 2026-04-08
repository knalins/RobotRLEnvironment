# Warehouse Robot Navigation — OpenEnv Environment

A real-world **warehouse robot navigation** environment built with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An LLM agent controls a delivery robot navigating a grid-based warehouse floor, avoiding obstacles to reach a delivery goal.

## Environment Description

The agent controls a robot (**R**) on a 2D grid and must navigate to a goal (**G**) while avoiding obstacles (**X**). The environment simulates autonomous warehouse robot path planning — a core challenge in logistics and robotics.

### Action Space

| Field | Type | Values |
|-------|------|--------|
| `direction` | `string` | `"up"`, `"down"`, `"left"`, `"right"` |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `grid` | `string` | Text rendering of the warehouse (R=robot, G=goal, X=obstacle, .=empty) |
| `robot_position` | `[row, col]` | Current robot coordinates |
| `goal_position` | `[row, col]` | Target delivery coordinates |
| `manhattan_distance` | `int` | Distance to goal |
| `message` | `string` | Feedback from last action |
| `step_count` | `int` | Current step number |
| `max_steps` | `int` | Maximum steps allowed |
| `task_name` | `string` | Current task difficulty |
| `score` | `float` | Current episode score (0.0–1.0) |

### Reward Structure

| Event | Reward | Description |
|-------|--------|-------------|
| Move closer to goal | `+0.1` | Encourages progress |
| Move away from goal | `-0.05` | Discourages regression |
| Wall collision | `-0.2` | Penalizes boundary hits |
| Obstacle collision | `-0.15` | Penalizes obstacle hits |
| Invalid direction | `-0.1` | Penalizes bad commands |
| Goal reached | `+1.0` | Major success reward |

### Task Score (Grader, 0.0–1.0)

- **Progress** (50%): How close the robot got to the goal relative to start
- **Goal bonus** (30%): Binary — did the robot reach the goal?
- **Efficiency** (20%): Fewer steps = higher score (only if goal reached)

## Tasks

| Task | Grid | Obstacles | Max Steps | Description |
|------|------|-----------|-----------|-------------|
| `easy` | 5×5 | 2 | 20 | Basic navigation with minimal obstacles |
| `medium` | 7×7 | 6 | 35 | Moderate density, requires planning |
| `hard` | 10×10 | 13 | 60 | Dense layout, multi-step planning needed |

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker
- Hugging Face CLI (`pip install huggingface_hub`)

### Local Development

```bash
# Install dependencies
pip install -e .

# Run the server locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run inference
export HF_TOKEN="your-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export SPACE_URL="http://localhost:7860"
python inference.py
```

### Docker

```bash
docker build -t warehouse-robot-nav .
docker run -p 7860:7860 warehouse-robot-nav
```

### Deploy to Hugging Face Spaces

```bash
huggingface-cli login
# Create a Space and push the code
```

## Validate Submission

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space .
```

## Architecture

```
MetaOpenEnv/
├── inference.py              # Baseline inference script
├── openenv.yaml              # Environment manifest
├── Dockerfile                # Container image
├── requirements.txt          # Dependencies
├── warehouse_robot_nav/      # Environment package
│   ├── models.py             # Action & Observation models
│   ├── environment.py        # Core environment logic
│   ├── client.py             # EnvClient for WebSocket
│   └── __init__.py
└── server/
    └── app.py                # FastAPI application
```
