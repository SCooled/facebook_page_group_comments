# Case authoring guide

Read `cases/S16.json` as the reference example and `core/models.py` for the exact schema
(`Case`, `CandidateVisible`, `Hidden`, `DeterministicCheck`). Validate with
`python -m runners.run_benchmark validate` from `automotive-agent-eval/`.

## Rules

1. **Filename = id.** `S01.json` ... `S24.json` for core scenarios, `A01.json` ... `A06.json` for adversarial variants.
2. **Two halves.** `candidate_visible` is everything the model sees. `hidden` is evaluator-only. Nothing in
   `candidate_visible` may reveal the hidden difficulty, the expected answer, or use evaluator vocabulary
   ("discrepancy", "trap", "the catch is", "note that the trim conflicts"). Describe the situation as a
   real buyer and real listings would present it. The model must find the problem itself.
3. **UK, October 2026.** Prices in GBP, UK plates (e.g. "71", "22"), UK terms (MOT, V5C, HPI, ULEZ/CAZ,
   VED, PCP, part exchange, keepers/owners, main dealer, independent, car supermarket). Mileage in miles.
   State "Today is 7 October 2026" in `scenario`.
4. **Realistic used prices for late 2026.** Sanity-check depreciation: roughly 15-25% off list in year one,
   then 10-15% a year, higher for premium and EV. If unsure, keep prices internally consistent across the
   case rather than precise to the market.
5. **Listings are structured objects** with stable ids (`L01`, `C1`, `A`, `B`) that the candidate can quote
   in `vehicle_ids`. Include at least: id, source, title, make, model, derivative, year, plate, mileage,
   fuel, transmission, body, seats, price_gbp, previous_owners, service_history, mot details where
   relevant, seller (type, location, distance_miles), days_listed, price_history where relevant,
   notable_equipment, description_excerpt. Put the realism in the details: MOT advisories, description
   wording, equipment lists that may or may not match the trim.
6. **Automotive accuracy.** Derivative, trim, engine and gearbox names must be real for the model and year
   (e.g. Kodiaq 1.5 TSI SE L DSG, Octavia 2.0 TDI vRS, Qashqai 1.3 DiG-T Tekna, 3 Series 320d M Sport
   Touring, Golf 1.5 eTSI Style DSG, Corolla 1.8 Hybrid Design Touring Sports, Tucson 1.6 T-GDi Premium,
   Sportage 1.6 T-GDi 3, XC60 B5 Inscription). Use the equipment that trim actually has. Do not invent
   recalls or faults; where a case needs a known issue, use well-documented ones (e.g. wet-belt on Ford
   1.0 EcoBoost, DQ200 DSG mechatronics, early N47 timing chain, PSA 1.2 PureTech belt, DPF on short-trip
   diesels, 12V battery drain on some EVs) and phrase them as the evidence would, not as a conclusion.
7. **`user_message`** is what the buyer actually types: one to three plain sentences.
8. **`hidden.critical_facts`**: 5-8 concrete facts a competent operator would identify, each specific
   enough to check the response against. Include the numbers.
9. **`hidden.expected_behaviours`**: 5-8 behaviours. **`hidden.critical_failures`**: 4-6 conditions that
   should each produce `critical_failure: true`.
10. **`hidden.escalation_expected`**: `required`, `acceptable` or `unnecessary`. Most intent, search and
    evaluation steps are `unnecessary`. Commitment steps with unresolved red flags are `required`.
11. **`hidden.deterministic_checks`**: 5-9 checks. Prefer checks on the structured block (`field_in`,
    `vehicles_include`, `vehicles_exclude`, `vehicles_subset`, `escalation`, `list_nonempty`,
    `figure_close`) over regex. For `mentions_any`, give 4-8 alternative patterns including numbers
    written both ways (`29[,.]?4`, `29400`). Regexes are case-insensitive. Mark a check `critical: true`
    with a `critical_failure_type` only where failing it is unambiguously a critical failure (e.g.
    recommending a vehicle over the hard budget, recommending `proceed` when decisive information is
    missing). Never mark a regex check critical.
12. **`hidden.expert_notes`**: 3-6 sentences in the voice of an experienced UK used-car operator: what they
    would conclude, what range or action is reasonable, and what would change their mind.
13. **`hidden.failure_categories_at_risk`**: 2-5 from the `FailureCategory` list in `core/models.py`.
14. **Journey cases** (`journey_id`, `journey_step`) carry the whole prior state in `conversation_state`
    as seeded facts, so each step stands alone. Keep ids, names, prices and dates consistent across steps.
15. **Adversarial variants** set `adversarial: true` and `variant_of: "S13"` etc. They are self-contained
    cases, not diffs. The manipulation (anchoring claim, scarcity, misleading title) must appear in
    `dealer_messages` or the listing as the dealer would actually word it.
16. JSON must be valid, UTF-8, 2-space indented, `£` written as the character.
