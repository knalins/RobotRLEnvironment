"""
Warehouse Robot Navigation — Inference Script
==============================================
Runs the LLM agent through all 3 tasks (easy, medium, hard).

Environment variables:
    API_BASE_URL       LLM API endpoint
    MODEL_NAME         Model identifier
    HF_TOKEN           Hugging Face / API key
    LOCAL_IMAGE_NAME   Docker image name (optional)
    SPACE_URL          HF Space URL for the environment
"""

import asyncio
import os
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

from warehouse_robot_nav import WarehouseRobotAction, WarehouseRobotEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
SPACE_URL = os.getenv("SPACE_URL", "https://openenv-warehouse-robot-nav.hf.space")

BENCHMARK = "warehouse_robot_nav"
TASKS = ["easy", "medium", "hard"]
TEMPERATURE = 0.3
MAX_TOKENS = 20

SYSTEM_PROMPT = textwrap.dedent("""
You are controlling a warehouse delivery robot on a grid.
R = your robot, G = goal/delivery point, X = obstacle, . = empty.
Navigate R to G while avoiding X.

Respond with EXACTLY one word: up, down, left, or right.
No explanation, no punctuation, just the direction.

Strategy: Move toward G. If blocked, go around the obstacle.
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r}", flush=True)


def ask_llm(client, grid, robot_pos, goal_pos, dist, msg):
    user_msg = (
        f"Grid:\n{grid}\n\n"
        f"Robot at row={robot_pos[0]}, col={robot_pos[1]}\n"
        f"Goal at row={goal_pos[0]}, col={goal_pos[1]}\n"
        f"Distance: {dist}\nFeedback: {msg}\n"
        f"Direction?"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip().lower()
        for d in ["up", "down", "left", "right"]:
            if d in text:
                return d
        return "right"
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "right"


async def run_task(client, env, task_name):
    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False

    # Set task via REST
    try:
        requests.post(f"{SPACE_URL}/set_task", json={"task_name": task_name}, timeout=10)
    except Exception:
        pass

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await env.reset()
        max_steps = obs.max_steps

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            direction = ask_llm(
                client, obs.grid, obs.robot_position,
                obs.goal_position, obs.manhattan_distance, obs.message
            )

            obs = await env.step(WarehouseRobotAction(direction=direction))
            reward = obs.reward or 0.0
            rewards.append(reward)
            steps = step

            log_step(step=step, action=direction, reward=reward, done=obs.done, error=None)

            if obs.done:
                break

        score = obs.score
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return score


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await WarehouseRobotEnv.from_docker_image(IMAGE_NAME)
    else:
        env = await WarehouseRobotEnv(base_url=SPACE_URL).__aenter__()

    try:
        all_scores = []
        for task_name in TASKS:
            score = await run_task(client, env, task_name)
            all_scores.append(score)
            print(f"[DEBUG] Task '{task_name}' score: {score:.2f}", flush=True)

        avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"\n[DEBUG] Average score: {avg:.2f}", flush=True)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())