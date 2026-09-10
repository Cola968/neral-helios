# Arctus 350M — 500M-token extended diagnostic results

Run: `34516677789`  
Checkpoint: `500,000,768` processed tokens, step `30,518`  
GPU: NVIDIA GeForce RTX 4070  
Suite: `arctus-500m-diagnostic-v1`

> This is a custom project diagnostic, not canonical NIS2 and not a public benchmark.

## Core measurements

- Fixed-seed held-out validation loss: **3.33349609375**
- Fixed-seed held-out perplexity: **28.0362**
- Held-out tokens evaluated: **131,072**
- 4-way continuation-preference accuracy: **53.57% (30/56)**
- Random 4-way chance level for reference: **25%**
- Exact instruction generation: **0.00% (0/10)**
- EOS rate over 30 generation probes: **0.00%**
- Mean repeated-trigram ratio: **63.70%**
- Median naive greedy decode speed: **23.48 BPE tok/s**

## Preference results by category

| Category | Correct | Accuracy |
|---|---:|---:|
| Code | 6/8 | **75.0%** |
| Factual | 5/8 | **62.5%** |
| Reasoning | 5/8 | **62.5%** |
| Science | 5/8 | **62.5%** |
| Bilingual | 4/8 | **50.0%** |
| Language | 4/8 | **50.0%** |
| Math | 1/8 | **12.5%** |

## Representative raw generations

### German continuation
Prompt: `Berlin ist eine Stadt, die`

Output begins:

` sich in der Nähe von Berlin befindet. Die Stadt ist ein Ort, an dem sich die Stadt befindet. ...`

Interpretation: grammatical local structure is emerging, but semantic redundancy and looping remain strong.

### English continuation
Prompt: `Berlin is a city that`

Output begins:

` is home to a number of important cultural and cultural sites. It is also home to a number of important cultural and cultural sites. ...`

Interpretation: fluent surface form is present, but repetition dominates longer continuation.

### Factual QA
Prompt: `Question: What is the capital of France?\nAnswer:`

Output begins:

` The capital of France is the capital of France. ...`

Interpretation: question/answer form is recognized, but the model does not reliably retrieve or emit the answer.

### Math
Prompt: `Problem: If x + 5 = 12, x =\nAnswer:`

Output begins:

` x = 12 ...`

Interpretation: open math generation is currently unreliable. The 4-way math preference score was only 12.5%.

### Reasoning
Prompt: `Anna is older than Ben, and Ben is older than Clara. Therefore`

Output contains:

`Anna is older than Clara.`

Interpretation: some relational reasoning signal is present, but the generation then repeats the conclusion.

### Code
Prompt: `def fibonacci(n):`

Output degenerates into repeated code fences. Despite that, code continuation-preference reached 75%, indicating a substantial gap between latent continuation preference and free generation quality.

## Interpretation

The 500M checkpoint is measurably beyond random continuation selection in several domains and has recognizable German/English syntax. It is still not a usable chatbot. The largest current failures are exact instruction following, EOS/termination, repetition control, open math, and converting latent preferences into clean generated answers.

The strong code/factual/science/reasoning preference scores should be treated as early signals, not as mature capabilities: this is a small custom diagnostic and the model has only foundation pretraining so far.

## Next comparison point

Repeat the exact frozen `arctus-500m-diagnostic-v1` suite at 1B cumulative training tokens. The important deltas are:

- held-out loss/perplexity;
- math preference accuracy;
- exact-instruction accuracy;
- EOS rate;
- repeated-trigram ratio;
- raw QA/math/code generation quality.

Do not change this v1 suite before the 1B comparison; add new tests under a new version instead.
