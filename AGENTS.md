# AGENTS.md

## What this repo is
- Goal: ObjectNav task research built around Habitat-Sim / Habitat-Lab.
- Primary workflows: run inside Docker, iterate via VS Code, write outputs to `./outputs`.

## How to run (canonical)
### Dev container (interactive)
- Preferred entrypoint:
  - `./scripts/run_dev.sh bash`
  - `./scripts/run_dev.sh python scripts/sanity_check.py`

### Smoke tests (pick the closest)
- Habitat imports:
  - `./scripts/run_dev.sh python scripts/sanity_check.py`

## Environment & data
- Datasets are expected under `${DATA_DIR:-$PWD/datasets}` on host and mounted into the container.
- Outputs go to `${OUTPUT_DIR:-$PWD/outputs}` (mounted to `/outputs` in dev/train scripts).
- If GPU is required, assume CUDA is available in the container.

## Where to make changes
- New/maintained code goes under `src/objectnav/`.
- Prefer the newer modules:
  - `src/objectnav/sim/` for simulator/navmesh/agent utilities
  - `src/objectnav/utils/` for shared utilities (viz, rotations, etc.)
- Treat `src/objectnav/legacy/` as compatibility code: you must not touch it.

## Coding conventions
- Prefer small, composable functions.
- Add/keep type hints on functions and always place a docstring.
- Avoid hidden side effects at import time (use explicit init/apply functions).
- Keep randomness controlled (seedable) when adding new training/eval code.
- Always ask clarifying questions before writing any code if requirements are ambiguous.
- If a task requires changes to more than 3 files, stop and break it into smaller tasks first.
- Always respect and follow standard protocols/conventions of ML research projects/repos.

## Validation checklist (before finishing)
- Run `python -m compileall src` (or equivalent) if you touched Python modules.
- Run the relevant smoke script(s) from the “How to run” section.
- If you changed configs, ensure defaults still work with the Docker workflow.

## Output & artifacts
- Do not commit large artifacts under `outputs/` or datasets.
- If a script writes files, default to `outputs/` and make paths configurable.

## When instructions conflict
- Follow system/developer instructions first.
- If repo guidance conflicts with a user request, call it out and proceed with the user request.