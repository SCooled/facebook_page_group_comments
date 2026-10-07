# Addendum: economic framing

The original brief asked whether a general agent can do the work well enough. The sharper question
is whether a general agent is the most economically efficient way to do each part of the work at
production scale. A specialist system is justified even at equal answer quality if it is materially
cheaper, faster, more reliable or higher-throughput.

## Specialist advantage vector

A = (ΔQ, ΔC, ΔL, ΔR, ΔT): quality, cost, latency, reliability, throughput, always measured against
the strongest plausible general-agent baseline.

| Dimension | Measured as |
|---|---|
| Quality Q | weighted rubric score / 100, zeroed on critical failure |
| Cost C | model tokens × price + primitive cost, per task; also per safe completion and per object |
| Latency L | p50 and p95 end-to-end including primitive time |
| Reliability R | P(valid output and no critical failure); score std across repetitions alongside |
| Throughput T | objects in scope per second; tasks per hour at configured concurrency |

Utility U = w_q Q + w_c C' + w_l L' + w_r R + w_t T' with C', L', T' mapped to [0,1] by saturating
transforms around configurable reference scales. Weight profiles in `config.yaml`:
`consumer_assistant` (quality and reliability dominate), `dealer_bulk` (cost and throughput
dominate), `balanced`.

Economic specialist advantage ESA = U(implementation) − U(general baseline), per provider, per
primary cognitive primitive, per weight profile. A specialist component is justified when ESA is
materially positive at expected production volume.

## Three implementations

| | Implementation | In the harness |
|---|---|---|
| A | general agent alone | `general` |
| B | general agent + lightweight domain context | `automotive_context` |
| C | general agent delegating to specialised primitives | `delegated`: filter, duplicate detection, ranking and comparable valuation engines pre-process the evidence; the model reasons over their output |

The reference primitives in `core/primitives.py` are deliberately simple and transparent. They
establish whether delegation moves the frontier at all; production primitives (learned ranker,
valuation model on transaction data, entity resolver over a vehicle ontology) replace the function
bodies without changing the accounting.

## Domain versus computational specialisation

An automotive agent may not need to understand cars better than a frontier model. The value may come
from subsystems that perform recurring automotive primitives more efficiently. The benchmark
therefore reports, per primitive, whether specialisation is warranted by better intelligence (ΔQ),
better economics (ΔC, ΔL, ΔT), better reliability (ΔR), or none.

## Revised thesis

Find high-volume cognitive primitives where a general agent is unnecessarily expensive, slow,
variable or operationally inefficient, and where a specialised system delivers a superior economic
frontier. See `PRIMITIVES.md` for the map.
