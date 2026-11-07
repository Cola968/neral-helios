# NERAL THINK: exact Python tracing without executing source code

Status: opt-in local THINK verifier. CPU-only, no external API, no Docker dependency, no untrusted Python execution and no change to Qwen or LoRA weights.

## Problem addressed

A local model can incorrectly predict a simple Python program's printed integer. A plain second generation repeats the same uncertainty. The new module neral_agent_python_trace.py uses Python's AST parser purely as data, interprets a tightly limited subset with explicit resource ceilings, and verifies exactly one printed integer. The supplied source code is never evaluated or executed. This works even when the Windows PC does not have Docker installed.

The module handles bounded integer arithmetic, variables, lists, tuples, dictionaries, safe indexing and slicing, nested finite range loops, conditionals, continue, and selected pure operations (sum, len, count, dictionary values). No imports, user-defined functions, arbitrary attribute access, subprocesses, file operations, global imports, network operations, eval or exec are supported. Unsupported inputs deliberately return no oracle.

The program must fit within 3000 characters, 180 AST nodes, 1200 interpreter steps, 128 elements per sequence or loop, 32 variables, 20 expression/statement nesting levels and a bounded integer magnitude. It returns None for all unsupported or ambiguous cases, never a guessed result.

## THINK integration

On a fresh single-turn request containing a complete Python fenced code block and an explicit question about printed output, NERAL:

1. makes its ordinary first model call;
2. checks any explicit final numeric answer against the exact local AST result;
3. on a mismatch, makes at most one model retry within existing THINK constraints;
4. if the model is still wrong, reports the deterministic tool result with model_generated=false and tool_used=neral_python_trace_static.

This is a genuine improvement to the **assistant system's** ability to answer supported code-tracing questions; it is not improvement to the v6 adapter weights and is not evidence that NERAL solves unrelated coding problems. A correct first model answer is reported as model_generated=true and verified=true. No trusted oracle is used on multi-turn conversations to avoid overriding contextual user instructions.

## Smoke examples

Supported:

What does this Python program print?

    values=[3,7,11,15,19]
    print(sum(values[1::2]))

The tool verifies integer 22 without executing that code.

Unsupported (must abstain rather than execute): any import, file/network access, function definition, comprehension, recursion, while loop, huge range, arbitrary print statement or unrecognized question.

## Tests and release policy

Run:

python -m unittest discover -s tests -p test_neral_agent_python_trace.py -v
python -m unittest discover -s tests -p test_neral_think.py -v
python -m unittest discover -s tests -p test_neral_think_api.py -v

The GitHub Actions THINK CPU workflow runs these tests. No GPU job, training or release promotion is triggered.

The historical 72-case v6 regression remains in quarantine. This module does **not** adjust its frozen benchmark scores. Any future claims of improvements require fresh, independent, paired end-to-end testing against v5/Qwen and tracking latency, tool-fallback rate and unverified answers.
