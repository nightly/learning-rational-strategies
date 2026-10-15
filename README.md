# Learning Rational Strategies for Temporal Objectives in Multi-Agent Systems

Code for our experiments from our paper 'Learning Rational Strategies for Temporal Objectives in Multi-Agent Systems'.

The repository covers multi-agent learning with probabilistic finite-trace temporal logic objectives. Each agent learns from its own probabilistic temporal objective, evaluation measures objective satisfaction and searches for profitable unilateral deviations.

If you use our code or theoretical results from our paper in your own work, please use the following citation:
```bibtex

```

This release includes the eight benchmark tasks, IPPO, recurrent IQL, their PR2 and counterfactual experience replay (CER) variants, handcrafted-reward baselines, and tabular Nash-Q for Stag Hunt.

## Install

Use [uv](https://docs.astral.sh/uv/) and Python 3.11 on Linux:

```bash
uv sync --locked
```

The default installation uses CPU JAX. For an NVIDIA GPU with a compatible
CUDA 12 driver, install the optional extra and select the GPU when running:

```bash
uv sync --locked --extra cuda
uv run --locked --extra cuda python -m src.reproduce classic_2 --platform gpu
```

`pygambit` supplies the Nash-Q equilibrium solver. If no wheel is available for your platform, its source installation needs a C++ compiler. Environment
rendering uses Pygame; the experiment runner needs no display. The Graphviz system executable is needed only optionally to render exported automaton diagrams.

## Run

Experiments run from plain JSON configurations and a command-line entry point.

First check training, checkpoint saving, satisfaction evaluation, and deviation search with a small integration run:

```bash
uv run --locked python -m src.reproduce stag_hunt --smoke --output exports_smoke
```

Then run a benchmark with its committed training budget and five seeds:

```bash
uv run --locked python -m src.reproduce classic_2 --algorithms ippo ippo_cer
```

Available tasks are `classic_2`, `capture_2`, `classic_3`, `capture_3`, `coins`,
`stag_hunt`, `harvest`, and `territory`. Algorithm names and all options are
listed by:

```bash
uv run --locked python -m src.reproduce --help
```

Use `--dry-run` to inspect the resolved input configuration, `--output PATH` to
choose a results root, and `--seed`, `--num-runs`, or `--steps` for controlled
changes. `--overrides FILE.json` merges scalar options or fields in the
`ippo`, `iql`, and `nashq` blocks. Existing profile outputs are protected, so
choose a fresh results root to rerun the same task and algorithm.

Each profile writes checkpoints, training histories, `satisfaction_report.json`,
`nash_report.json`, and timing records under `exports/<task>/<algorithm>/`.
The task directory also records the invocation and probability thresholds.
The smoke preset is solely an installation check; its estimates are not
meaningful experimental results.

## License

Licensed under [Apache License 2.0](LICENSE).
