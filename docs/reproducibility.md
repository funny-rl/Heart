# Reproducibility and performance evidence

## Determinism

Record the following for a reproducible result:

- HEART version and exact Git commit;
- environment ID and every rule override;
- JAX and jaxlib versions;
- backend and device model;
- reset key or seed;
- policy name/configuration and policy-key derivation;
- action sequence for a reported semantic failure.

Reset is deterministic for a fixed key, software stack, and JAX behavior.
Built-in policies use explicit keys only for tie-breaking. Reusing a constant
policy key at every turn is allowed but is not equivalent to splitting a key.

## Batched execution

The authoritative environment operates on one state. Use `jax.vmap` for batch
dimensions and `jax.jit` for compilation. A typical fixed deal uses
`jax.lax.scan` for 52 actions. Do not introduce host callbacks, rendering, or
file I/O into the compiled scan.

Keep actions as `jnp.int32` inside long-lived compiled rollout boundaries.
Changing an argument's dtype or shape changes JAX's abstract signature and can
trigger a separate compilation.

The current dense state layout is deliberate: it keeps rules and replay data
easy to audit. Bit-packed hands or compressed history should be considered only
when a measured target workload is limited by state memory; such a change is a
public-state compatibility decision, not a default micro-optimization.

## Benchmark receipt

The harness separates environment work from policy work and safe validation
from the trusted fast path:

```bash
python benchmarks/random_rollout.py --batch-size 4096 --runs 10 \
  --workload engine-only --step-mode safe
python benchmarks/random_rollout.py --batch-size 4096 --runs 10 \
  --workload engine-only --step-mode trusted
python benchmarks/random_rollout.py --batch-size 4096 --runs 10 \
  --workload policy-inclusive --policy hard --step-mode safe
```

`engine-only` uses deterministic first-legal actions and excludes policy PRNG.
`policy-inclusive` accepts `random`, `easy`, `medium`, or `hard`. Both return a
small final-state checksum instead of materializing a 52-step output trajectory;
all timed dispatches are explicitly synchronized. Batch size and run count must
be positive.

Report both deals/s and card-actions/s together with:

```text
commit:
HEART/JAX/jaxlib versions:
hardware and backend:
workload, policy, and step mode:
batch size and runs:
compile/warm-up excluded: yes/no
command:
deals/s:
card actions/s:
```

Do not compare numbers across different rule graphs, hardware, precision,
batch sizes, or warm-up treatment without saying so. A passing correctness
suite does not establish strategic realism or policy quality.
