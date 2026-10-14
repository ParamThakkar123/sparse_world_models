# Sparse Mechanics World Models — Research Program v3

*Prepared 2026-10-12. Amends `sparse_mechanics_world_models_research_program_v2.pdf` (v2).
v2 stays the statement of ambition; this document is the plan we execute. Where they
disagree, this one wins, and every disagreement is listed in §1 with its reason.*

---

## 0. Core thesis and central claim

**Thesis (unchanged from v2).** Learn a compact library of reusable state-transition
mechanisms that explains how interactive worlds change after player actions: predict a
sparse, causally grounded state patch rather than a dense next frame.

**The one falsifiable claim the first paper must establish (sharpened):**

> A shared sparse mechanism model predicts *which* state components change, *why*, and
> *by how much* — on change **onsets**, not just continuations — better than (i) the best
> trivial rule, (ii) dense and persistence baselines, and (iii) per-game models, on
> held-out games, at matched data and compute; and it reaches a target accuracy on a new
> game with a smaller adaptation budget than training from scratch.

Every word in that claim maps to a measurement in §5. If a word cannot be measured at a
game's integration tier (§3), the claim is not made for that game.

---

## 1. What changed from v2, and why

| # | Amendment | Reason |
|---|---|---|
| A1 | **The trivial-rule battery is a required baseline and part of every exit criterion.** v2's baselines are persistence/dense only. | Our ICLR work: across three benchmarks the best one-liner scored F1 0.93–0.96 and beat every learned model, including five published ones. Games are worse — "the player entity changes", "moving things keep moving", "things near the player change" are strong by construction. |
| A2 | **Report change-mask metrics split into onset vs continuation.** Onset = a component at rest that starts changing (collision start, damage, spawn, door opens). | In our tabletop data the onset split is where learned models genuinely beat trivial rules (contact gate 0.29–0.41 vs trivial 0.00); the aggregate mostly measures inertia. |
| A3 | **Mechanism labels come from engine events, never from the shape of the state diff.** | If `c_t` is defined by which variables changed, it is free given `m_t` and mechanism F1 is circular. |
| A4 | **Procgen demoted to the pixel track.** OCAtari + Crafter (+ stable-retro, ViZDoom) become the structured-state core. | Procgen exposes no usable entity state, so it cannot supervise `m_t`. OCAtari gives object-centric Atari state; Crafter exposes its world. |
| A5 | **Counterfactual pairs move from Phase C into Phase B for tier-3 environments.** | ALE `cloneState/restoreState`, emulator save-states and Crafter world copies make exact branching nearly free. W3 (gate as causal mask) is already our strongest causal result. |
| A6 | **Data access is organised into four integration tiers (§3); every result is reported with its tier, and claims are restricted to what the tier can support.** | Most v2 metrics (mask P/R, mechanism F1, impossible-transition rate, unchanged-vs-unobserved, RQ5) cannot be measured from video + actions alone. |
| A7 | **Hooking is built only on existing mod frameworks.** We own a thin per-game exporter and one game-agnostic protocol/harness; we never write memory hacks or engine injection. A game that needs custom low-level hooking stays at tier 0–1. | Reverse engineering is not research, breaks on every patch, and raises EULA/anti-cheat risk. |
| A8 | **Genre generality (v2 Phase VI) is built mainly from tier-3 research platforms; AAA games are the scale/realism stress test, not the main evidence.** | Tier-3 platforms give exact state and branching across genres cheaply; AAA gives neither. |
| A9 | **In AAA games, measure the run-to-run noise floor before any counterfactual claim.** | No exact save-states; replayed scripts diverge on their own. A counterfactual difference only counts above that floor. |
| A10 | **Transfer is reported as an adaptation-cost curve and a per-mechanism transfer table**, alongside zero/few-shot numbers. | "Does it transfer" is not binary; the cost of adaptation and *which* mechanisms transfer is the informative result. |
| A11 | **Compute is serialised.** Capture and training never run concurrently. | 16 GB machine; concurrent runs have killed pipelines and even torch's DLL load. |

Smaller v2 fixes: the §3.2 formula renders as raw LaTeX; the Appendix A table columns
overlap; verify the SCOPE arXiv ID, dataset licences (CrossFPS is CC BY-NC 4.0) and mod
tool version support before citing them.

---

## 2. Architecture

### 2.1 Model: one shared core, thin per-game adapters

```
 per game (adapter)            shared (core)                         per game (adapter)
 ──────────────────            ─────────────                         ──────────────────
 observation → state    ──▶    gate m_t (what changes)        ◀──    native controls →
 (telemetry map, or            mechanism c_t (why)                   shared action vocabulary
  vision estimator)            delta Δ_t (how much)
                               ApplyPatch(s_t, m_t, Δ_t)
```

* The core starts from the existing sparse residual model (`models/sparse_gating.py`,
  `sparse_residual.py`, `delta_heads.py`). New: the mechanism head `c_t`, and typed
  entities with variable attribute sets (today's `StateLayout` assumes fixed `POSE_DIM`).
* Adapters are small and their parameter/data budgets are always reported.
* In video-only games the core runs over learned slots instead of true entities, and the
  claim weakens from "the right *entities* change" to "the right *regions/slots* change".
  Papers say so explicitly.

### 2.2 Data layer: `games/` package and the WorldBridge protocol

One game-agnostic interface. The model and evaluation code never know which game they
are on.

```
games/
  schema.py        # GameState: entities (typed, attribute dict, observed flag), relations,
                   #   globals, player/controller state; Action (shared vocabulary);
                   #   MechanismEvent (shared taxonomy + "game_specific" overflow)
  bridge.py        # WorldBridge API: reset(task) / step(action) / snapshot() / restore()
                   #   / state() / events(); snapshot/restore optional by tier
  collect.py       # rollouts → episode-tagged .npz (existing format) + audit_battery format
  branch.py        # counterfactual pairs: snapshot, branch one action/timing/param, roll both
  noise_floor.py   # repeat identical scripts, measure outcome spread (tiers 1–2)
  adapters/
    ocatari.py  crafter.py  retro.py  vizdoom.py       # tier 3
    gtav.py  vicecity.py  cyberpunk.py                  # tiers 0–2, talk to mods/ over a socket
mods/
  gtav/            # ScriptHookVDotNet (C#) state exporter + scene reset (or a DeepGTAV fork)
  vicecity/        # plugin-sdk (C++) exporter
  cyberpunk/       # Cyber Engine Tweaks (Lua) exporter
```

* Actions to AAA games go through a virtual gamepad (ViGEmBus + `vgamepad`) so they are
  genuine controller inputs. Frames via `dxcam`/OBS. Every record carries frame id and a
  QPC timestamp; alignment error is measured, not assumed.
* Metadata per episode: game build, mod versions, settings, FPS, input device, task seed,
  reset method, known nondeterminism.
* Splits are by whole episode / session / route / scene, using the `build_clean_splits.py`
  approach (we have already been bitten by chunk-level leakage once).

### 2.3 Shared vocabularies (v0, expected to grow)

* **Actions:** move(x,y), look(x,y), jump, crouch, throttle, brake, steer, interact,
  use/fire, enter/exit, menu-noop.
* **Mechanisms:** motion-response, contact/collision, damage/health, resource change
  (ammo/inventory/money/stamina), spawn/despawn, destruction, state toggle (doors, switches),
  timer/cooldown, attach/detach (grasp, vehicle entry), game-specific.
  The growth of the "game-specific" class across games is itself a generality measurement.

---

## 3. Integration tiers and what each can claim

| Tier | Data | Examples |
|---|---|---|
| 0 | Video + actions | CrossFPS, any game |
| 1 | + HUD reading (health, speed, wanted stars, minimap) | Most AAA titles |
| 2 | + state export through an existing mod framework | GTA V, Vice City, Cyberpunk (partial) |
| 3 | + full state + exact save-states | OCAtari/ALE, Crafter, stable-retro, ViZDoom, open-source games |

| Claim / metric | T0 | T1 | T2 | T3 |
|---|---|---|---|---|
| Next-frame / latent prediction, pixel drift, efficiency | ✅ | ✅ | ✅ | ✅ |
| Transfer & adaptation curves on prediction metrics | ✅ | ✅ | ✅ | ✅ |
| Planning task success (needs a success check) | — | ✅ | ✅ | ✅ |
| State prediction error | pseudo | partial | ✅ | ✅ |
| Change-mask P/R, onset F1, false-unchanged rate | — | — | ✅ | ✅ |
| Mechanism/event F1 and timing | — | — | ✅* | ✅ |
| Counterfactual consistency | approx. (save-load + replay) | approx. | above noise floor | exact |
| Impossible-transition rate | — | partial | ✅ | ✅ |
| Unchanged vs unobserved | — | — | ✅* | ✅ |
| RQ5 / telemetry ablation | — | — | ✅ | ✅ |

\* only for the variables and events the mod actually exposes.

---

## 4. Phases, deliverables and gates

Phases advance on evidence, not dates. Each has an exit criterion; each gate has a
pre-agreed action on failure so a negative result is reported, not rationalised.

### Phase 0 — Close out the current paper (prerequisite)
Confirm ICLR 2027 submission status and reserve machine time for the rebuttal window.
Nothing below may compete with it for the machine.

### Phase A — Harness, schema, and the degeneracy check
**Deliverables:** `games/schema.py`, `bridge.py`, `collect.py`, `branch.py`; OCAtari and
Crafter adapters; episode-level splits; `audit_battery.py` extended with game rules
(`player_entity`, `already_changing`, `near_player`, `periodic_timer`, `same_as_last_step`,
plus `--extra-rules` invented against each new game).

**Gate G1 (go/no-go for RQ1 as written):** run the battery on game transitions *before
training anything*.
* Pass: the best trivial rule leaves headroom on onset F1 → proceed with mask claims.
* Fail: one-liners ≥ ~0.93 on onsets too → redefine RQ1 around counterfactual differences
  and mechanism prediction before building the model. Report the result either way.

### Phase B — Sparse mechanism model on tier 3, with counterfactuals
**Deliverables:** typed-entity `StateLayout`; mechanism head; training on per-step data
*and* exact counterfactual pairs from save-states; baselines (no-op, linear/AR, dense
RSSM-style, our existing sparse residual unchanged, transformer where affordable,
trivial-rule battery); ablations (no mechanism head, no mask, no counterfactual pairs,
counterfactual vs random augmentation, sparsity strength, entities vs latent-only).

**Exit:** beats the trivial ceiling and dense/persistence on onset F1 and state error at
matched data/compute, ≥5 seeds; counterfactual training improves changed-state P/R over
non-counterfactual training; mechanism F1 above the label-frequency baseline.

### Phase C — Genre suite and cross-game transfer (tier 3)
**Suite (small, one per genre to start):** platformer (stable-retro), FPS (ViZDoom),
open world (Crafter), arcade (OCAtari, 3–5 games with different event structure);
racing (SuperTuxKart/TORCS) and strategy/sim (microRTS or Factorio) if Phase B passes.

**Evaluations:** held-out games; leave-one-genre-out; zero-shot vs few-shot adapters vs
per-game models vs shared dense; adaptation-cost curves (0/10/100/1000 episodes);
per-mechanism transfer table.

**Exit:** shared core beats per-game and shared-dense on held-out games with the
adaptation budget reported; at least one mechanism class demonstrably transfers across
genres, and the ones that don't are named.

### Phase D — Pixel track and state estimation
**Deliverables:** the same tier-3 environments rendered to pixels; a vision state estimator
(slots/detector) feeding the core; state-estimation error measured against true state;
CrossFPS (tier 0) as an external pixels-only transfer test.

**Exit:** the oracle-state vs estimated-state gap is measured and attributed — model error
and perception error reported separately (our tabletop pixel run was blocked by
perception; we expect the same failure mode here and need it quantified).

### Phase E — GTA V pilot (tier 2) — engineering may start during Phase B
Building the bridge uses no training compute, so it can run in parallel; *collection*
cannot overlap training.

**Deliverables:** pinned game build with updates off; exporter (DeepGTAV fork if it still
works on the current build, else a small ScriptHookVDotNet script); harness integration;
one empty-road driving task, 20 repeats.

**Exit (= v2 Phase IV):** stable repeatable task; measured timestamp alignment; measured
noise floor; documented state-label quality. Only then expand the task suite (driving,
pedestrian/vehicle interaction, camera/movement, persistent state) and collect.

### Phase F — External AAA validation
Vice City (plugin-sdk; original PC 1.0 edition only) and Cyberpunk 2077 (CET; player state
first, other entities only if reliably exposed). Frozen core + fitted adapters; separate
appearance transfer from dynamics transfer; report failures and tool/version limits.

### Phase G — Generality report
Which mechanisms transfer, which do not, and the adaptation cost per game and per genre.

---

## 5. Metrics (v2 table, amended)

| Family | Metrics | Amendment |
|---|---|---|
| State prediction | position/velocity error, discrete accuracy, identity switches | report oracle-state and estimated-state separately |
| Sparse updates | mask P/R, update sparsity, false-unchanged rate | **split onset vs continuation; always shown beside the best trivial rule** |
| Mechanism | event F1, mechanism accuracy, timing error | labels from engine events only; compare to label-frequency baseline |
| Counterfactual | changed-state P/R, downstream event agreement | AAA: difference must exceed measured noise floor |
| Long horizon | error vs horizon, impossible-transition rate, drift | — |
| Transfer | zero/few-shot on held-out games and genres | **adaptation-cost curve + per-mechanism transfer table** |
| Planning | task success, regret vs oracle, recovery | oracle/scripted/random anchors, as in `planning_mpc.py` |
| Efficiency | FLOPs, latency, memory, updates/step | sparsity alone is not evidence of efficiency |

Statistics: ≥5 seeds for anything in a claim (n=3 floored every p-value at 0.25 in the
ICLR work); claim ledger in the style of `paper/claims.py`.

---

## 6. Per-game onboarding checklist

1. Assign an integration tier; record why.
2. State mapping → `GameState` (or slot estimator at tier 0–1).
3. Action mapping → shared vocabulary.
4. Mechanism mapping: engine events → shared taxonomy, unmatched → `game_specific`.
5. Task scripts: fixed start states, success criteria.
6. Noise floor: repeat identical scripts (tiers 0–2).
7. Trivial-rule battery on that game's transitions, including rules invented against it.
8. Episode/session/route/scene-level splits.
9. Licence/EULA check; raw assets stay local; release code, schemas, task definitions and
   permitted annotations only.

---

## 7. Risks added to v2's table

| Risk | Mitigation |
|---|---|
| Game benchmarks are degenerate like ours were | Gate G1 before any model work; onset split; battery in every exit criterion |
| Mechanism labels circular | Engine events only; label-frequency baseline |
| Bridge breaks on a game patch | Pin builds, disable updates, record versions; fall back to tier 0–1 |
| Typed-entity generalisation of `StateLayout` destabilises existing results | New code path; existing tabletop tests must stay green |
| Compute contention | Serialise; capture and training never overlap |
| Scope creep across too many games | Phase C starts with one game per genre; add only after its exit |

---

## 8. Immediate next actions

1. Confirm ICLR status and rebuttal timing (Phase 0).
2. Install and smoke-test OCAtari and Crafter alongside the existing `ale-py` 0.11.2; verify
   `cloneState/restoreState` determinism (same branch twice → identical state).
3. Write `games/schema.py` + `bridge.py` with tests; OCAtari adapter first, Crafter second.
4. `collect.py` emitting both the existing `.npz` format and the `audit_battery` format.
5. Add game rules to `audit_battery.py`; run Gate G1 on 3–5 OCAtari games and Crafter.
6. Write the G1 result into `experiments/RESULTS.md` before any training.

## 9. Open decisions

* Which OCAtari games for G1 — proposal: Pong, Breakout, Freeway, Seaquest, MsPacman
  (spanning contact, spawn, resource and timer mechanics).
* Whether Phase E bridge engineering starts during Phase B (recommended) or after Phase C.
* GPU/VRAM headroom for running GTA V capture on this machine (not yet measured).
