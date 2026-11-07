# NERAL-EXT MBPP functional scoring (exploratory)

This is a CPU-only regrade of the original 400-case GPU run
https://github.com/Cola968/neral-helios/actions/runs/42395097203 .
It is not a new independent benchmark, not training, and not GPU inference.

Exactly 50 Stratum-A cases are MBPP Python functions. The old scorer gave
all 50 cases the status ungraded, even if Python syntax was correct.
Incomplete model answers were stored as empty strings and cannot be restored.

The new module scripts/neral_ext_mbpp_grader.py takes the exact original
saved answers and the pinned manifest containing the public MBPP assertions.

Each candidate must contain a complete standalone Python program or a single
Python code fence. Syntax checking with ast.parse happens on the host, but
NEVER execution. Candidate code, test setup and assertions execute only
inside an already installed python:3.11-alpine Docker container, with:
network disabled, read-only root and mounted inputs, uid 65534, 128 MB memory,
0.5 CPU, 32 PID ceiling, ALL Linux capabilities dropped, no-new-privileges
and 10 seconds wall-time timeout per case. A completion marker after all
assertions is required; exit status zero alone cannot count as a pass.

If Docker Desktop is not running with Linux containers or the required image
is missing, the manual workflow refuses to run and does not execute Python
source on Windows as a fallback.

Per variant, the statuses are: passed, failed, timeout, unavailable, censored,
no_code, invalid_python and reserved bad_contract. These remain separate:
a censored or ungradable response is NOT a tested coding failure. A timeout
is not a pass. Any unavailable sandbox disqualifies the aggregate export.

The workflow is .github/workflows/neral-ext-mbpp-cpu-manual.yml. It requires
manual authorization on main and is serialized with the existing local-GPU
research jobs. Only the SHA-pinned summary.json counts are uploaded as an
artifact. Private prompts, code, tests and model answers stay on the user's
Windows research folder. There is no public raw-data upload.

Tests (offline CPU only):
python -m unittest discover -s tests -p test_neral_ext_mbpp_grader.py -v
python -m unittest discover -s tests -p test_neral_ext_mbpp_workflow.py -v

Passing the public MBPP unit tests is evidence of passing those particular
tests, not proof of secure programs, unseen pretraining generalization or
general coding intelligence. Do not combine coding and numeric denominators
into a single headline accuracy without identifying censored or ungraded cases.
No adapter promotion or release-quality claim is allowed.
