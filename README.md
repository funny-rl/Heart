<div align="center">

<img src="docs/assets/logo.svg" alt="HEART — accelerator-native Hearts for competitive multi-agent reinforcement learning" width="640">

# HEART

**An accelerator-native Hearts environment for competitive multi-agent reinforcement learning.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/) [![JAX](https://img.shields.io/badge/JAX-%E2%89%A50.4.38-orange.svg)](https://github.com/jax-ml/jax) [![Version](https://img.shields.io/badge/version-0.1.0-2563eb.svg)](#release-status) [![Status](https://img.shields.io/badge/status-research--preview-f59e0b.svg)](#release-status) [![Citation](https://img.shields.io/badge/cite-CITATION.cff-lightgrey.svg)](CITATION.cff)

[Why HEART](#why-heart) · [Rules](#the-environment) · [Install](#installation) · [Quickstart](#quickstart) · [Play](#playing-against-the-policies) · [Leaderboard](docs/leaderboard.md) · [Batching](#batched-rollouts) · [Rendering](#rendering-and-replays) · [Contracts](#core-contracts) · [Documentation](#documentation)

<a href="docs/assets/rendering/classic-v0-deal-transition.html"><img src="docs/assets/rendering/classic-v0-deal-transition.gif" alt="HEART classic-v0 left-pass deal transition from player 0's seat" width="760"></a>

<sub>`classic-v0` · left passing through the next-deal boundary · 57 seat-view frames in 7.92 seconds · the interactive HTML replay is <a href="docs/assets/rendering/classic-v0-deal-transition.html">in the repository</a> — GitHub shows it as source, so download the raw file and open it locally.</sub>

</div>

---

HEART is a compact JAX-native implementation of the card game Hearts for
multi-agent learning, evaluation, and reproducible simulation. The Python
distribution is `heart-marl`; the installed package is imported as `heart`.

Its deal engine is designed to compose directly with `jax.jit`, `jax.vmap`, and
`jax.lax.scan`. `classic-v0` is a complete match: passing, thirteen-trick
deals, and a running score to 100 points. Rules and state transitions remain in
the lean compiled path, while rendering and replay expansion remain host-side.

> [!NOTE]
> HEART 0.1.0 is a research preview. No GitHub Release or PyPI package has been
> published yet, and the public API may evolve before a stable release.

## Why HEART

| Contract                | What it provides                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| **Structured horizons** | Every deal has 52 card plays, and four pass events before them except on hold deals.       |
| **Pure JAX transition** | Immutable array state, explicit random keys, and no hidden environment mutation.           |
| **Competitive rewards** | Zero-sum deal outcomes, delivered at every deal boundary.                                  |
| **Legal-action masks**  | A 52-way play mask enforces card rules; passing adds a 286-way combination mask.           |
| **Reference opponents** | Seeded `easy`, `medium`, and `hard` rule policies share the learning-policy interface.     |
| **Inspectable replays** | Seat-view or omniscient terminal, HTML snapshot, and portable interactive replay outputs.  |
| **Human play**          | One seat played by a person against rule tiers or plugged-in learned policies.            |
| **Leaderboard entries** | Policies submitted as portable graphs, checked against a signature and a compute budget.  |

Training algorithms are intentionally not bundled. HEART supplies the game
contract and reference policies; callers retain ownership of batching,
learning, evaluation, and experiment tracking.

## The environment

`heart.make()` returns `classic-v0`, a complete match with standard scoring.
One deal of Hearts also had its own published ID once; nothing was built on it,
so there is one contract now rather than two.

| Rule              | Value                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------ |
| Players           | 4 independent players                                                                      |
| Deck              | Standard 52-card deck, 13 cards per player                                                  |
| Passing           | Left, right, across, hold; repeating every four deals                                       |
| Pass action       | One of `13C3 = 286` unordered triples of sorted hand slots                                  |
| Deal events       | 56 on passing deals; 52 on hold deals                                                       |
| Opening lead      | 2♣                                                                                          |
| Following suit    | Required when possible                                                                      |
| Leading hearts    | Forbidden until hearts are broken, unless only hearts remain                                |
| First trick       | Point cards cannot be discarded when a non-point card is available                          |
| Heart penalty     | 1 point each                                                                                |
| Q♠ penalty        | 13 points                                                                                   |
| Shooting the moon | Shooter scores 0 for the deal; every opponent scores 26                                     |
| Match end         | Checked after a deal when any cumulative score reaches 100                                  |
| Winners           | Every player tied for the lowest cumulative score; no tie-break deal                        |
| Reward timing     | Zero within a deal; a normalized zero-sum vector at each deal boundary                      |

The lowest effective score wins. At a deal boundary, player `i` receives

```text
reward_i = (mean(effective_scores_of_other_players) - effective_score_i) / 26
```

For an ordinary deal the four rewards sum to zero mathematically (subject to
ordinary float32 rounding); raw penalties do not. The denominator is the
configured total point value (`13 + queen_of_spades_penalty`). A moon shot uses
a dedicated winner-takes-all signal instead: the shooter receives `0` and each
opponent receives `-1`.

See the normative [`classic-v0` contract](docs/classic.md) for phase masks,
reward semantics, the single-learner adapter, and deal-boundary state, and the
[deal rules](docs/rules.md) for the trick semantics every deal follows.

## Installation

Python 3.10, 3.11, and 3.12 are targeted. Until a formal release exists,
install from a checked-out revision:

```bash
git clone <repository-url> HEART
cd HEART
python -m pip install -e .
```

For development checks:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

JAX chooses CPU, GPU, or TPU from the installed runtime. Follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
accelerator-specific wheels.

## Quickstart

Run a complete match with separate pass and play policies:

```python
import jax

import heart

env = heart.make("classic-v0")
pass_policy = heart.make_rule_pass_policy("medium")
play_policy = heart.make_rule_policy("hard")
key = jax.random.key(0)
state, observation = env.reset(key)

while not bool(state.terminated):
    key, action_key = jax.random.split(key)
    if int(observation.phase) == heart.PASS:
        action = pass_policy(observation, action_key)
    else:
        action = play_policy(observation.game, action_key)
    state, observation, rewards, terminated, info = env.step(state, action)

print("match scores:", state.match_scores)
print("winners:", state.winner_mask)
```

A play action is an integer card ID in `[0, 52)`; a pass action is one of the
286 triples. Floats, booleans, out-of-range IDs, masked actions, and
post-terminal actions are invalid: they leave state unchanged, return zero
rewards, and set `info.invalid_action=True`.

## Playing against the policies

```bash
python -m heart                                      # three medium opponents
python -m heart --seat easy --seat medium --seat hard
python -m heart --human-seat 2 --port 8080 --seed 7
```

The command prints a local URL. Opening it puts you in one seat of a
`classic-v0` match: you see your own hand, the other three are face down, and
the environment checks every move against the same masks the compiled path
uses. Opponents play one at a time at a speed you pick, a finished trick is
swept to the seat that won it, and the match can be reset at any point.

A seat is a packaged rule tier — `easy`, `medium`, `hard` — or a
`package.module:factory(argument)` reference returning a `SeatPolicy`, which is
how a trained checkpoint joins the table without becoming a dependency of this
package. See the [human play contract](docs/play.md).

## Leaderboard

Policies are submitted as portable computation graphs, so any architecture is
welcome and the host never imports submitter code. Entries are checked against a
fixed signature, a size limit, and a per-decision compute budget, then seated
against each other on shared deals and ranked on penalty points per deal, with
an Elo beside it and an error on both.

Entries arrive as a pull request adding one directory under
[`submissions/`](submissions/), and CI checks the file before review.

**→ [Submission rules and standings](docs/leaderboard.md)**

## Train one player

Use the single-learner adapter when one learned policy should face rule-based
opponents:

```python
import jax
import jax.numpy as jnp

env = heart.make_single_agent(
    controlled_player=0,
    opponents=("easy", "medium", "hard"),
)
state, observation = env.reset(jax.random.key(0))

for _ in range(13):
    slot = jnp.argmax(observation.action_mask)  # shape: (13,)
    state, observation, reward, terminated, info = env.step(state, slot)
```

The 13 actions are stable slots for the player's initially dealt cards. The
adapter masks played or illegal slots and automatically runs the other three
players until the learner acts again. The underlying rules engine still applies
all 52 card plays, while the learner sees exactly 13 decisions and one scalar
terminal reward.

For a complete classic match, select the versioned mode and optionally separate
the pass and play opponents:

```python
env = heart.make_single_agent(
    "classic-v0",
    controlled_player=0,
    pass_opponents="medium",
    play_opponents="hard",
)
```

A passing deal exposes one pass decision plus 13 plays, for 14 learner
decisions; a hold deal exposes 13 plays. The adapter records the core events
compressed inside each decision and expands them for replay with
`env.replay_events(before_state, info)`. For learning across the sparse
deal-boundary rewards, `gamma=1` is recommended.

Use the core `heart.make()` environment for four-policy MARL or self-play; it
keeps global 52-card actions so every policy and replay shares one unambiguous
card encoding.

Measure thousands of simultaneous learner-versus-rules games with:

```bash
python benchmarks/single_agent_rollout.py --batch-size 4096 --runs 10 \
  --opponents medium
```

## Batched rollouts

HEART does not introduce a separate vector-environment abstraction. Standard
JAX transformations batch the same authoritative transition:

```python
import jax
import jax.numpy as jnp

import heart

env = heart.make("classic-v0")
keys = jax.random.split(jax.random.key(0), 4096)
batch_reset = jax.jit(jax.vmap(env.reset))
batch_step = jax.jit(jax.vmap(env.step))

states, observations = batch_reset(keys)
actions = jnp.argmax(observations.pass_action_mask, axis=-1)
states, observations, rewards, terminated, infos = batch_step(states, actions)
```

When actions have already been selected from the matching authoritative mask,
trusted training code may use `env.step_unchecked`. It skips current-action
validation but keeps the same transition and next observation:

```python
trusted_batch_step = jax.jit(jax.vmap(env.step_unchecked))
```

Passing an unvalidated, non-integer, out-of-range, or illegal action to this
fast path has undefined behavior. Keep `env.step` at untrusted boundaries.

Only the active player's action is supplied for each environment. See
[`benchmarks/random_rollout.py`](benchmarks/random_rollout.py) for a compiled
52-action throughput harness. Use `--workload engine-only|policy-inclusive` to
separate transition and policy cost, and `--step-mode safe|trusted` to measure
validation overhead. Performance claims should report the source revision,
backend, hardware, batch size, warm-up, and measured run count.

## Core contracts

| Surface               | Entry point                                                            | Boundary                                                                |
| --------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Environment           | `heart.make`, `HeartEnv`, `ClassicEnv`                                 | Versioned single-deal and complete-match handles over immutable state   |
| State and observation | [`docs/environment.md`](docs/environment.md)                           | Omniscient environment state versus player-private policy input         |
| Rules and reward      | [`docs/rules.md`](docs/rules.md), [`docs/classic.md`](docs/classic.md) | Versioned legality, scoring, termination, passing, and reward semantics |
| Reference policies    | `heart.make_rule_policy`                                               | Mask-respecting, seeded baselines; not claims of optimal play           |
| Rendering and replay  | [`docs/rendering.md`](docs/rendering.md)                               | Seat-view and omniscient host output, excluded from the compiled path   |
| Reproducibility       | [`docs/reproducibility.md`](docs/reproducibility.md)                   | Explicit seeds, revision reporting, and benchmark receipts              |

The learning observation is intentionally player-private. Full observability is
a **rendering contract**, not permission for a policy to access hidden opponent
hands.

## Built-in opponents

```python
policy = heart.make_rule_policy("medium")  # easy, medium, or hard
action = policy(observation, key)
```

- `easy` ducks avoidable tricks and unloads dangerous point cards when void;
- `medium` additionally pressures opponents with safe early spade leads;
- `hard` also unloads high cards on the point-free opening trick and cashes a
  high forced winner when acting last on a clean trick.

Classic passing follows the same tier order: `easy` unloads intrinsically
dangerous cards, `medium` manages Q♠ exposure and retains low-spade cover, and
`hard` additionally creates a diamond void or, when 2♣ is not held, a club void.

These are reproducible baselines, not optimal Hearts agents or measured human
models. New policies should remain isolated from the environment transition.

## Rendering and replays

Render one unbatched match state as text or a standalone browser page:

```python
print(heart.render_classic_ansi(state, viewer=0))
heart.save_classic_html(state, "heart-game.html", viewer=0)
```

These show the cumulative scoreboard, pass direction and choices, and
completed-deal overlays. `render_ansi`, `render_html` and `save_html` do the
same for one deal on its own, which is what the match renderers draw each deal
with.

The checked-in preview follows one complete left-pass deal across its boundary:

![HEART classic-v0 deal transition](docs/assets/rendering/classic-v0-deal-transition.gif)

The 57-frame interactive HTML replay is
[checked in](docs/assets/rendering/classic-v0-deal-transition.html). GitHub renders
repository HTML as source, so download the raw file to open the player in a browser.

Create compact animated previews with the optional rendering dependency:

```bash
python -m pip install -e '.[render]'
python examples/render_classic.py --gif heart-preview.gif
```

Record a reset state and its successors to create an interactive replay:

```python
heart.save_classic_replay_html(
    states,
    "heart-replay.html",
    viewer=0,
    fps=2.0,
)
```

For the classic single-learner adapter, preserve the pre-decision state and
expand its compressed event trace before rendering:

```python
before = state
state, observation, reward, terminated, info = env.step(state, action)
classic_frames = env.replay_events(before, info)
heart.save_classic_replay_html(classic_frames, "classic-replay.html", viewer=0)
```

HTML replays provide play/pause, frame stepping, a timeline, speed controls,
and keyboard navigation. Every hand is visible. `viewer` only marks the local
player and rotates that seat to the bottom; `viewer=None` selects a neutral
spectator orientation. Generated outputs do not belong in source control. See
the [rendering contract](docs/rendering.md).

## Release status

| Surface                | Current state                                           |
| ---------------------- | ------------------------------------------------------- |
| Version                | `0.1.0`                                                 |
| Environment IDs        | `classic-v0`                                            |
| Source status          | Research preview                                        |
| GitHub tag and Release | Not published                                           |
| PyPI package           | Not published                                           |
| CI                     | Python 3.10–3.12 workflow enabled                       |
| API stability          | Pre-1.0; incompatible changes require changelog entries |

A formal release exists only when one validated source commit, an annotated
tag, a GitHub Release, and built distributions identify the same code. See
[`docs/release.md`](docs/release.md).

## Documentation

| Area                                               | Document                                             |
| -------------------------------------------------- | ---------------------------------------------------- |
| Documentation map                                  | [`docs/README.md`](docs/README.md)                   |
| State, observations, actions, and transitions      | [`docs/environment.md`](docs/environment.md)         |
| Deal rules, trick semantics, and rewards           | [`docs/rules.md`](docs/rules.md)                     |
| `classic-v0` match, passing, rewards, and learning | [`docs/classic.md`](docs/classic.md)                 |
| Seat-view and omniscient rendering and replay      | [`docs/rendering.md`](docs/rendering.md)             |
| Seeds, batching, and performance evidence          | [`docs/reproducibility.md`](docs/reproducibility.md) |
| Release and compatibility policy                   | [`docs/release.md`](docs/release.md)                 |

## Contributing and security

Issues and pull requests are welcome. Reproducible bug reports should include
the HEART and JAX versions, source revision, backend, mode, seed, and smallest
failing action sequence.

See [CONTRIBUTING.md](CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

## Citation

For this research preview, cite the exact commit used together with
[CITATION.cff](CITATION.cff).

```bibtex
@software{park_heart_2026,
  author    = {Park, Hyunwoo},
  title     = {HEART: An Accelerator-Native Hearts Environment for Competitive Multi-Agent Reinforcement Learning},
  year      = {2026},
  license   = {Apache-2.0},
  note      = {Research-preview source; cite the exact Git commit used}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright © 2026 Hyunwoo
Park. Developed at CIDA Lab, University of Seoul. See [NOTICE](NOTICE) for
attribution.
