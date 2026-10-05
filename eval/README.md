# Evaluation harness

Compares two Second-Me backends (for example upstream 2025 and this fork) on one frozen question set, scored by an LLM judge.

The question set, answers and scores from the published comparison are **not included**: they were generated from the author's private documents. To reproduce the method, generate a question set from your own uploaded documents.

## What it measures

| Category | What it tests | Metric |
|---|---|---|
| fact | Facts stated in the uploaded documents | Accuracy (1 / 0.5 / 0) |
| unanswerable | Questions the memory cannot answer; does the model invent facts? | Share of answers that don't invent |
| style | Voice, viewpoint and judgment compared with the person's real writing | persona 1–5, quality 1–5 |
| general | General capability | quality 1–5 |
| update | Add, edit and delete a fact without retraining | Accuracy per phase |

Every question runs in two modes:
- `rag`: L0 retrieval on. This is the primary metric.
- `bare`: retrieval off, which shows what the LoRA itself learned.

The harness also records latency and chat-template tag leaks.

## Workflow

```bash
export EVAL_GATEWAY=http://127.0.0.1:3425/v1/chat/completions   # any OpenAI-compatible endpoint
export EVAL_JUDGE_MODEL=<judge model>  EVAL_GEN_MODEL=<generator model>

python gen_evalset.py <path/to/lpm.db> evalset.json         # once; freeze the output
python run_eval.py    http://localhost:8002 old evalset.json answers.jsonl
python run_eval.py    http://localhost:8003 new evalset.json answers.jsonl
python update_test.py http://localhost:8002 old evalset.json answers.jsonl
python update_test.py http://localhost:8003 new evalset.json answers.jsonl
python judge.py score   evalset.json persona_ref.txt answers.jsonl scores.jsonl
python judge.py compare evalset.json persona_ref.txt answers.jsonl old new rag pairs.jsonl
python judge.py report  scores.jsonl pairs.jsonl
```

`persona_ref.txt` holds excerpts of the person's own writing. The judge uses it to decide whether an answer sounds like them.

Each backend must already be serving its trained model (`POST /api/kernel2/llama/start`). `update_test.py` uploads two fictional facts, edits them and then deletes them, so the instance is left as it was.

## Pass criteria (fixed before running)

In `rag` mode, all of the following must hold:
- Fact accuracy is not lower than the baseline.
- The non-invention rate on unanswerable questions is not lower than the baseline.
- The persona score is higher than the baseline.
- The new version wins more blind A/B comparisons than it loses.
- There are zero template-tag leaks.

## Caveats

- The same model family generated the questions and judged the answers, so blind A/B comparisons are the primary signal and absolute scores are secondary.
- The retrieval threshold strongly affects results. Calibrate `L0_SIMILARITY_THRESHOLD` for your embedding model and use the same value for both backends.

## Results

See the evaluation section of the top-level [README](../README.md).
