# Human expert responses (Condition C)

One Markdown file per case: `S01.md`, `S16.md`, `A03.md` ...

The expert answers the same candidate payload the models receive. Print it with:

    python -c "from core.cases import *; from pathlib import Path; print(build_candidate_payload(load_cases(Path('cases'), ['S16'])[0]))"

The expert must not see `hidden` before answering. Finish the answer with the same JSON block the
models produce (see `prompts/output_format.txt`), so deterministic checks and escalation metrics apply.

Run and score exactly like a model:

    python -m runners.run_benchmark run   --batch b1 --providers human --configs general --reps 1
    python -m runners.run_benchmark score --batch b1

The judge is blind to provider, so human answers are scored on the same footing.
