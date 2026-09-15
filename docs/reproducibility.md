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

## Benchmark receipt

Run the included random legal-policy harness:

```bash
python benchmarks/random_rollout.py --batch-size 4096 --runs 10
```

Report both deals/s and card-actions/s together with:

```text
commit:
HEART/JAX/jaxlib versions:
hardware and backend:
batch size and runs:
compile/warm-up excluded: yes/no
command:
deals/s:
card actions/s:
```

Do not compare numbers across different rule graphs, hardware, precision,
batch sizes, or warm-up treatment without saying so. A passing correctness
suite does not establish strategic realism or policy quality.
