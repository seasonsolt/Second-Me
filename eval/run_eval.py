"""Ask every eval question to a running Second-Me backend and record the answers.

Usage: python run_eval.py <backend_base_url> <label> <evalset.json> <out.jsonl> [--modes rag,bare]
  rag  = enable_l0_retrieval on  (primary: how users actually chat)
  bare = retrieval off           (what the LoRA itself has absorbed)
The llama-server with the trained model must already be running on that backend.
"""
import argparse
import json
import time

import requests


def ask(base: str, question: str, rag: bool) -> dict:
    body = {
        "messages": [{"role": "user", "content": question}],
        "stream": True,
        "temperature": 0.1,
        "max_tokens": 800,
        "metadata": {"enable_l0_retrieval": rag, "enable_l1_retrieval": False},
    }
    t0 = time.time()
    ttft = None
    parts = []
    error = None
    with requests.post(f"{base}/api/kernel2/chat", json=body, stream=True, timeout=600) as r:
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                err = chunk["error"]
                error = err.get("message") if isinstance(err, dict) else str(err)
                break
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            if delta and ttft is None:
                ttft = time.time() - t0
            parts.append(delta)
    return {
        "answer": "".join(parts),
        "error": error,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "latency_s": round(time.time() - t0, 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("label")
    p.add_argument("evalset")
    p.add_argument("out")
    p.add_argument("--modes", default="rag,bare")
    p.add_argument("--categories", default="fact,unanswerable,style,general")
    a = p.parse_args()

    items = [it for it in json.load(open(a.evalset)) if it["category"] in a.categories.split(",")]
    try:
        done = {(r["label"], r["mode"], r["id"]) for r in map(json.loads, open(a.out)) if r.get("label")}
    except FileNotFoundError:
        done = set()
    with open(a.out, "a") as f:
        for mode in a.modes.split(","):
            for it in items:
                if (a.label, mode, it["id"]) in done:
                    continue
                res = ask(a.base, it["question"], rag=(mode == "rag"))
                rec = {"label": a.label, "mode": mode, "id": it["id"], "category": it["category"], **res}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{a.label}/{mode}] {it['id']} {res['latency_s']}s {'ERR ' + res['error'] if res['error'] else ''}")


if __name__ == "__main__":
    main()
