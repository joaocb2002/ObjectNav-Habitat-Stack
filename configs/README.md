# Configs

This project uses a Hydra-style configuration layout.

- `config.yaml` is the root composition file (shared `paths` + `hydra.run.dir`).
- `sim/`, `navmesh/`, `grid_map/` are config groups.
- `experiment/` contains reproducible experiment presets (compositions).
- `runs/` contains ad-hoc developer run presets (non-reproducible, local overrides).

Why have `runs/` if we already have `experiment/`?
- `experiment/` is for versioned, reproducible presets (papers/benchmarks).
- `runs/` is for quick local overrides (debugging, profiling) without polluting experiment configs.

Precedence (highest to lowest):
1. CLI overrides
2. YAML composition (`configs/*.yaml`)
3. Code defaults (structured configs / dataclasses)
