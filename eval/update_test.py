"""Memory update test: add / modify / delete a fact without retraining.

Usage: python update_test.py <backend_base_url> <label> <evalset.json> <answers.jsonl>
For each `update` item:
  add     -> upload add_fact, index it, ask      (expect: answer)
  modify  -> replace with modified_fact, ask     (expect: modified_answer)
  delete  -> remove the file, ask                (expect: no longer stated as known)
Answers are appended with ids like `update-057@add` and an `item_override` so judge.py
grades each phase against the right expectation.
"""
import json
import sys
import time
import urllib.parse

import requests

from run_eval import ask


def upload(base: str, filename: str, text: str) -> int:
    r = requests.post(f"{base}/api/memories/file", files={"file": (filename, text.encode(), "text/markdown")}, timeout=600)
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(f"upload failed: {body}")
    doc_id = (body.get("data") or {}).get("document_id")
    if doc_id is None:
        docs = requests.get(f"{base}/api/documents/list", timeout=60).json()["data"]
        doc_id = max(d["id"] for d in docs if d.get("name") == filename)
    return doc_id


def index(base: str, doc_id: int) -> None:
    for path in ("/api/documents/chunks/process", f"/api/documents/{doc_id}/chunk/embedding"):
        r = requests.post(f"{base}{path}", timeout=600)
        r.raise_for_status()
        if r.json().get("code") not in (0, None):
            raise RuntimeError(f"{path} failed: {r.text[:300]}")


def delete(base: str, filename: str) -> None:
    r = requests.delete(f"{base}/api/memories/file/{urllib.parse.quote(filename)}", timeout=120)
    r.raise_for_status()


def main(base: str, label: str, evalset: str, out: str) -> None:
    items = [it for it in json.load(open(evalset)) if it["category"] == "update"]
    with open(out, "a") as f:
        for it in items:
            filename = f"eval-{it['id']}.md"
            phases = []
            doc_id = upload(base, filename, it["add_fact"])
            index(base, doc_id)
            phases.append(("add", {"expected": it["answer"]}))
            res = ask(base, it["question"], rag=True)
            f.write(json.dumps(_rec(label, it, "add", res, phases[-1][1]), ensure_ascii=False) + "\n")

            delete(base, filename)
            doc_id = upload(base, filename, it["modified_fact"])
            index(base, doc_id)
            res = ask(base, it["question"], rag=True)
            f.write(json.dumps(_rec(label, it, "modify", res, {"expected": it["modified_answer"]}), ensure_ascii=False) + "\n")

            delete(base, filename)
            time.sleep(1)
            res = ask(base, it["question"], rag=True)
            f.write(json.dumps(_rec(label, it, "delete", res, {"category": "unanswerable"}), ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{label}] {it['id']} done")


def _rec(label: str, it: dict, phase: str, res: dict, override: dict) -> dict:
    return {"label": label, "mode": "rag", "id": f"{it['id']}@{phase}", "category": "update",
            "item_override": override, **res}


if __name__ == "__main__":
    main(*sys.argv[1:5])
