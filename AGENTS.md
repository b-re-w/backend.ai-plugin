# AGENTS.md

Guidance for AI coding agents (and humans) working in this workspace.

## Workspace layout

| Path | Role | May you edit it? |
|---|---|---|
| `backend.ai/` | Upstream Backend.AI core (open source, LGPLv3). Checked out for **reference only**. | **No.** Read it to learn plugin APIs. |
| `backend.ai-webui/` | The user's **fork** of the Backend.AI WebUI, local `develop` branch. | **Yes**, for lab features that the WebUI plugin mechanism cannot reach (e.g. extra rows in the session detail screen). Keep changes small and self-contained so they survive upstream merges. |
| `backend.ai-plugin/` | **Our code.** Fractional-GPU and per-model accelerator plugins, plus the GPU idleness monitor for spot lending, for the lab cluster. | Yes. All work happens here. |

`backend.ai/` stays reference only: never patch the core, only plug into it.

Before touching code in `backend.ai-plugin/`, read, in order:
1. `backend.ai-plugin/docs/INTENT.md`: why this exists and what "done" means.
2. `backend.ai-plugin/docs/SPEC.md`: the behavioural contract. **SPEC is the source of truth.**
3. `backend.ai-plugin/README.md`: install / operate.

## Principles (in priority order)

1. **Owner first.** A GPU owner's session must never lose work because of lending. Spot workloads are
   always the victim: they get evicted first, OOM-killed first, and throttled first. When a rule would
   hurt an owner to help a spot session, the rule is wrong.
2. **Fail safe, not fail open.** When the monitor is unsure (NVML error, unknown process on the GPU,
   cannot map a PID to a container, stale data), it must *not lend* and must *reclaim* what is lent.
   Never assume a GPU is idle because a probe failed.
3. **Never enforce silently-unenforced limits.** If fractional GPU limits cannot actually be enforced
   (e.g. the HAMi-core library is missing), degrade loudly (log at ERROR, fall back to whole-GPU
   allocation) instead of handing out "0.5 GPU" that is really a whole shared GPU.
4. **Do not modify the Backend.AI core.** Everything plugs in through public extension points: the
   `backendai_accelerator_v21` entry-point group, etcd plugin config, Docker labels, and the agent's
   hook-library mechanism. If something seems to require patching `backend.ai/`, stop and write it up
   in SPEC under "Open questions" instead.
5. **Small, observable, reversible.** Every automated action (lend, reclaim, move) is logged
   with the reason and the measurements that triggered it. Operators must be able to turn lending off
   per node or per GPU without restarting Backend.AI.

## Hard constraints

- **Licensing.** Do not decompile, reverse engineer, or copy code from Lablup's proprietary
  "Virtual CUDA API" / fGPU enterprise plugin. Build only from open components: Backend.AI OSS
  (LGPLv3, consumed via entry points), HAMi-core (`libvgpu.so`, Apache-2.0, installed separately, not
  vendored), NVIDIA NVML via `nvidia-ml-py`.
- **Version drift.** The lab ran Backend.AI **25.15.6** and is moving to the latest stable release, **26.8.3**.
  `backend.ai/` and `backend.ai-webui/` are on a local `develop` branch pinned to **26.8.3** and
  **v26.8.1** (`main` is left at the upstream head); the E2E runs 26.8.3. Plugin signatures differ
  across versions (e.g. `device_alloc` is a `Mapping[SlotName, Mapping[DeviceId, Decimal]]` in 25.x
  and typed as `DeviceAllocation` in newer code), so code must accept both shapes. Anything not checked
  on the version the lab will run must be flagged in SPEC's verification table.
- **Plugin key collision.** `cuda_frac` uses device key `cuda` and slot `cuda.shares` so the WebUI
  treats it as fGPU. It cannot coexist with the stock `cuda` (cuda_open) plugin on the same agent.
  Agent allow/block lists match **module-path prefixes**, not entry-point names, which is why each
  entry point we want to block independently lives in its own module.
- **Custom device keys need agent config.** Every per-model key (`pro6000`, ...) must be listed in the
  agent's `[resource] allocation-order`, and each device's `device_name` must equal the plugin key;
  both failures only show up when a session is actually created, so re-run the WSL E2E after touching
  device registration.
- **The dev environment is not a lab node.** Development happens on Windows; `backend.ai-plugin/e2e/`
  runs a real Backend.AI manager + agent in WSL2 with Docker Desktop, a fake-NVML mode that mimics the
  lab servers, and the laptop's RTX 4050. WSL2 cannot run HAMi-core, does not report per-process NVML
  data, and hides container PIDs, so those paths still need a native Linux GPU node. Keep hardware- and
  Docker-touching code in thin adapter modules and decision logic pure and unit tested. Never claim a
  hardware path "works" unless it ran on real hardware; mark it `UNVERIFIED` in SPEC's table instead.
- **Security scope.** Assumed environment is a trusted research lab.
- **Spot is a Backend.AI launch mode.** Spot sessions are real Backend.AI sessions started from the
  WebUI session launcher. Never run spot workloads outside Backend.AI (plain `docker run`, a side
  queue). The design is not approved yet: do not build it until the user approves it.

## Working rules

- Change SPEC first, then code, then tests, in the same change. If code and SPEC disagree, the code is
  the bug unless SPEC is updated deliberately.
- Code, identifiers, comments, logs, and `AGENTS.md` are in English. User-facing docs
  (`README.md`, `docs/INTENT.md`, `docs/SPEC.md`) are in Korean for lab members; keep identifiers,
  config keys, and CLI names in English inside them.
- Python ≥ 3.12, type hints everywhere, `dataclasses` for plain data, no global mutable state.
- Tests: `pytest` under `backend.ai-plugin/tests/`. Pure modules must be importable without
  `ai.backend.*`, `pynvml`, or Docker being installed.
- Do not add runtime dependencies beyond what SPEC lists without updating SPEC.
- Use `decimal.Decimal` for anything that becomes a Backend.AI resource slot amount; never floats.
- **Commits.** Commit `backend.ai-plugin/` and `backend.ai-webui/` work at natural checkpoints (a
  SPEC + code + tests change that passes), on each repo's local `develop` branch; do not push unless
  asked. Commits are authored by the user only: never add `Co-Authored-By` or any other trailer crediting an AI assistant or tool.
- **No em dashes (U+2014).** Do not use the em dash anywhere in `backend.ai-plugin/`, in our changes to
  `backend.ai-webui/`, or in this file:
  code, comments, logs, docs, examples, commit messages. Use a colon, comma, period, or parentheses
  instead. Check with `python backend.ai-plugin/scripts/check_no_em_dash.py` before finishing a change.
