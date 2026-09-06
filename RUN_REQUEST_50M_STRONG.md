# Neral-50M strong training run

Run after the RawMath loss-weighting correction. Pipeline: frozen 20M/50M baseline → complete document-bounded Stage B → identical benchmarks → 15M-token RawMath Stage C with explicit 35/65 math/language loss weighting → identical benchmarks again.

Weights and private corpus remain local to the self-hosted runner.
