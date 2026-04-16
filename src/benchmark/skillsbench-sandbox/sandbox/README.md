# SkillsBench Sandbox Quickstart

## 1) Introduction

This repository exposes SkillsBench tasks as HTTP environments for LLM-agent RL rollouts.

- Original repo: `./skillsbench` — upstream: https://github.com/benchflow-ai/skillsbench (do not modify)
- Sandbox implementation: `./sandbox`, `sandbox/start_sandbox_server.py`
- Rollout example: `./examples/agent_rollout.py`

## 2) Clone Repository

```bash
git clone git@github.com:langfengQ/skillsbench-sandbox.git
cd skillsbench-sandbox
git submodule update --init
```

## 3) Create Virtual Environment

```bash
conda create -n "skillsbench-sandbox" python=3.12
conda activate skillsbench-sandbox
pip install openai gymnasium pyyaml
```

## 4) Start Sandbox Server

```bash
python3 sandbox/start_sandbox_server.py --config ./configs/sandbox_config.yaml
```

Health check:

```bash
curl -s http://127.0.0.1:8080/health
```

## 5) Call Sandbox API

### Create environment

```bash
curl -s http://127.0.0.1:8080/envs \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"tasks","task_id":"xlsx-recover-data"}'
```

### Get task instruction

```bash
curl -s http://127.0.0.1:8080/envs/<env_id>/instruction
```

### Execute one step

```bash
curl -s http://127.0.0.1:8080/envs/<env_id>/step \
  -H 'Content-Type: application/json' \
  -d '{"command":"ls -la /root","timeout_sec":60}'
```

### Evaluate reward

```bash
curl -s http://127.0.0.1:8080/envs/<env_id>/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"timeout_sec":1200}'
```

### Delete environment

```bash
curl -s -X DELETE http://127.0.0.1:8080/envs/<env_id>
```

Delete environment **and its Docker image**:

```bash
curl -s -X DELETE "http://127.0.0.1:8080/envs/<env_id>?remove_image=true"
```

### Delete a Docker image directly

Useful when the env was already deleted but the image remains:

```bash
curl -s -X DELETE http://127.0.0.1:8080/images/<image_tag>
```

## 6) Rollout Examples

### Rule agent (quick pipeline check)

```bash
python3 examples/agent_rollout.py \
  --server-url http://127.0.0.1:8080 \
  --dataset tasks \
  --task-id xlsx-recover-data \
  --agent rule \
  --max-steps 1 \
  --evaluate-every-step
```

### OpenAI agent

```bash
export OPENAI_API_KEY=<your_api_key>
python3 examples/agent_rollout.py \
  --server-url http://127.0.0.1:8080 \
  --dataset tasks \
  --task-id powerlifting-coef-calc \
  --agent openai \
  --model gpt-5.3-codex \
  --history-turns 20
```

If you want the rollout script to auto-start the server, add: `--auto-start-server`.

Defaults now follow task metadata:

- Max rounds inferred from difficulty (Core/Extended/Extreme style): `10/30/50`
- Step/eval timeout inferred from `task.toml` (`[agent].timeout_sec`, `[verifier].timeout_sec`)
- Context management uses an 8K-token sliding window (oldest turns dropped)
- With Skills mode is on by default: OpenAI-skills-style metadata (`name`, `description`, `path`) is injected in user observation context, and the agent loads `SKILL.md` on demand

Useful overrides:

```bash
--max-steps 20
--step-timeout-sec 900
--eval-timeout-sec 1200
--context-window-tokens 8000
--disable-skill-injection
```

## 7) Prompt Template

- `examples/skillbench_universal_prompt.md`

`examples/agent_rollout.py` uses this template style by default for `--agent openai`:

- System prompt defines command-only contract and execution policy.
- Per-step prompt carries instruction + latest execution feedback (`stdout`/`stderr`/exit code/reward).

This is designed for cross-task use and can be further specialized per task family if needed.
