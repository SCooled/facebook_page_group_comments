# Cognitive primitive map for the automotive purchase journey

The benchmark decomposes the transaction into recurring cognitive primitives rather than only
journey stages, because primitives recur across verticals and are the unit at which a specialised
implementation can be economically justified. Each case is tagged with the primitives it exercises
(`primitives`), the one whose economics it is designed to expose (`primary_primitive`) and the prior
to be tested (`expected_class`: general / hybrid / specialist). Results are aggregated on those tags.

## Primitives

| Primitive | Definition | Automotive instances | Volume / shape at production | Prior | Candidate specialist implementation |
|---|---|---|---|---|---|
| retrieve | fetch candidate objects from a large corpus | listing search across platforms, spec lookup, comparable pull, MOT/HPI fetch | very high, 10^3–10^6 objects per user task, predictable schema | specialist | indexed search over normalised listings; tool call, not model reasoning |
| filter | drop objects that fail hard constraints | budget, gearbox, fuel, seats, body, distance, CAZ compliance | very high, trivially deterministic | specialist | rule engine over structured fields (reference: `hard_constraint_filter`) |
| rank | order candidates by expected utility to this user | shortlist from 20–1,000 listings, dealer offer ranking | high, needs user weights, repeated as constraints change | specialist | learned ranker over structured attributes + user weights; model explains the top-k (reference: `heuristic_ranker`) |
| classify | assign an object to a category with consequences | segment from use case, risk level, escalation needed, keeper pattern benign/concerning, dealer answer evasive/adequate | medium; mixes structured and conversational input | general for judgement, hybrid for structured | model for conversational judgement; classifier where inputs are structured |
| extract | pull structured facts from unstructured text | equipment from descriptions, service events from invoices, dealer answers to questions, missing fields | high, repetitive, well defined | specialist | small-model or schema-guided extraction; frontier model unnecessary |
| normalise | resolve entities and canonical identities | derivative/trim identification, duplicate listing detection, model-year vs registration, equipment-to-trim consistency | very high; errors cascade into valuation and search | specialist | vehicle ontology + entity resolver (reference: `duplicate_detector`) |
| compare | weigh two or more objects across attributes | A vs B trade-offs, bundled offers, substitution candidates | medium; needs user utility | hybrid | structured diff from specialist; narrative and weighting by model |
| estimate | produce a quantity with uncertainty from evidence | fair-price range, finance total cost, warranty value, adjustment for spec/mileage | high; variance and precision matter; data-dependent | specialist | valuation model on transaction data; finance calculator (reference: `comparable_valuation`) |
| detect_anomaly | flag objects or claims inconsistent with expectations | trim mismatch, stale listing, odd mileage pattern, unsupported dealer claim, removed FSH wording | high; cheap on structured data, hard on prose | hybrid | rules + outlier detection over listing history; model for claim-vs-evidence reasoning |
| predict | forecast behaviour | next price cut, seller flexibility, days to sale | medium; needs proprietary listing-history data | hybrid | model on historical listing changes and outcomes |
| optimise | choose an action under constraints | opening/target/walk-away, substitution under budget, requirement relaxation | medium; strategy benefits from outcome data | hybrid | negotiation policy from outcomes; model frames and communicates |
| communicate | elicit, explain, persuade, hand off | requirement elicitation, trade-off explanation, dealer messaging, escalation to a human | every task; conversational | general | frontier model |

## What the harness measures per primitive

For every (provider, configuration) and every primary primitive: Q quality, C cost per task and per
object, L p50/p95 latency, R reliability, T objects per second, and U under each weight profile.
`esa_by_primitive.csv` reports the economic specialist advantage of the context and delegated
implementations against the general baseline. A primitive is a candidate for specialised
implementation when ESA is materially positive at the weight profile of the intended product, or
when the general agent's cost per object makes production volume uneconomic even at equal quality.

## Expected result classes

- **General-agent territory**: communicate, conversational classify (S01–S04, S19, S20, S24, A03, A05).
- **Specialist primitives**: retrieve, filter, rank, normalise, extract, estimate on structured data
  (S05, S06, S08, S13, S16, S17, S22, A01, A02, A04).
- **Hybrid**: compare, detect_anomaly on claims, predict, optimise (S07, S09–S12, S14, S15, S18, S21, S23, A06).

The tags are priors. The benchmark exists to confirm or refute them with the measured vector.

## Scale cases

The 30 cases operate on 2–20 objects, which exposes quality but understates throughput and cost
effects. To measure the search/rank frontier at production scale, add scale variants of S05 and S06
with 200–1,000 synthetic listings (the generator is a natural next step), run them under `general`
and `delegated`, and compare cost per object and objects per second. That is where implementation C
is expected to dominate regardless of quality.
