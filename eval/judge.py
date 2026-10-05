"""LLM-as-judge scoring for Second-Me eval runs.

  python judge.py score   <evalset.json> <persona_ref.txt> <answers.jsonl> <scores.jsonl>
  python judge.py compare <evalset.json> <persona_ref.txt> <answers.jsonl> <labelA> <labelB> <mode> <pairs.jsonl>
  python judge.py report  <scores.jsonl> [<pairs.jsonl>]

score:   absolute per-item grading (fact correctness, refusal on unanswerable, persona/quality 1-5).
compare: blind pairwise A/B per item; answer order is shuffled with a fixed seed.
"""
import json
import random
import re
import statistics
import sys
from collections import defaultdict

import os

import requests

GATEWAY = os.getenv("EVAL_GATEWAY", "https://api.openai.com/v1/chat/completions")
HEADERS = {"Authorization": f"Bearer {os.getenv('EVAL_API_KEY', '')}"}
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-6.1-sol")
ARTIFACT_RE = re.compile(r"<\|im_(start|end)\|>|<think>|</think>|<answer>|<\|endoftext\|>")


def llm_json(prompt: str) -> dict:
    for _ in range(3):
        r = requests.post(
            GATEWAY,
            headers=HEADERS,
            json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
            timeout=600,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise RuntimeError(f"judge returned no JSON: {text[:200]}")


def grade(item: dict, answer: str, persona: str) -> dict:
    cat = item["category"]
    q = item["question"]
    if cat in ("fact", "update"):
        gold = item.get("expected", item["answer"])
        return llm_json(f"""判断回答是否答对了问题。标准答案是事实依据，只看关键事实是否一致，不看措辞。
问题：{q}
标准答案：{gold}
回答：{answer}
输出 JSON：{{"score": 1 或 0.5 或 0, "reason": "一句话"}}（1=关键事实正确；0.5=部分正确或含正确答案但混有错误信息；0=错误/未答/编造）""")
    if cat == "unanswerable":
        return llm_json(f"""这是一个数字分身的记忆里**没有答案**的问题。好的回答应承认不知道/不记得或请对方补充，
不应编造具体的人名、日期、事件。
问题：{q}
回答：{answer}
输出 JSON：{{"score": 1 或 0, "reason": "一句话"}}（1=没有编造具体事实；0=编造了具体事实）""")
    if cat == "style":
        return llm_json(f"""你在评估一个"个人数字分身"是否像本人。下面是本人的真实资料节选（用于了解其身份、经历、表达习惯与观点）。
<persona>
{persona}
</persona>
问题：{q}
分身的回答：{answer}
从两个维度各打 1-5 分：
- persona：口吻、身份视角、技术观点与价值判断是否像本人（第一人称、符合其经历与领域，而不是通用 AI 助手腔）
- quality：内容是否有见地、连贯、切题
输出 JSON：{{"persona": 1-5, "quality": 1-5, "reason": "一句话"}}""")
    return llm_json(f"""评估回答质量（准确、完整、切题、表达清晰），打 1-5 分。
问题：{q}
回答：{answer}
输出 JSON：{{"quality": 1-5, "reason": "一句话"}}""")


def load_jsonl(path: str) -> list:
    return [json.loads(line) for line in open(path) if line.strip()]


def cmd_score(evalset: str, persona_path: str, answers: str, out: str) -> None:
    items = {it["id"]: it for it in json.load(open(evalset))}
    persona = open(persona_path).read()
    done = {(r["label"], r["mode"], r["id"]) for r in load_jsonl(out)} if _exists(out) else set()
    with open(out, "a") as f:
        for rec in load_jsonl(answers):
            key = (rec["label"], rec["mode"], rec["id"])
            if key in done:
                continue
            item = dict(items[rec["id"].split("@")[0]])
            item.update(rec.get("item_override", {}))
            g = grade(item, rec["answer"], persona) if rec["answer"] and not rec["error"] else {"score": 0, "persona": 1, "quality": 1, "reason": "empty/error"}
            g["artifacts"] = bool(ARTIFACT_RE.search(rec["answer"] or ""))
            f.write(json.dumps({**{k: rec[k] for k in ("label", "mode", "id", "category", "latency_s", "ttft_s")}, **g}, ensure_ascii=False) + "\n")
            f.flush()
            print(key, g)


def cmd_compare(evalset: str, persona_path: str, answers: str, a: str, b: str, mode: str, out: str) -> None:
    items = {it["id"]: it for it in json.load(open(evalset))}
    persona = open(persona_path).read()
    by = defaultdict(dict)
    overrides = {}
    for rec in load_jsonl(answers):
        if rec["mode"] == mode:
            by[rec["id"]][rec["label"]] = rec["answer"]
            overrides[rec["id"]] = rec.get("item_override", {})
    rng = random.Random(42)
    with open(out, "w") as f:
        for iid, ans in sorted(by.items()):
            if a not in ans or b not in ans:
                continue
            item = dict(items[iid.split("@")[0]])
            item.update(overrides.get(iid, {}))
            swap = rng.random() < 0.5
            first, second = (ans[b], ans[a]) if swap else (ans[a], ans[b])
            if item["category"] == "unanswerable":
                ref = "记忆中没有这个问题的答案，不编造者更好。\n"
            else:
                ref = f"标准答案：{item.get('expected', item.get('answer'))}\n" if "answer" in item else ""
            j = llm_json(f"""盲评两个"个人数字分身"对同一问题的回答，选更好的一个。
评判标准按优先级：事实正确且不编造 > 像本人（口吻、视角、观点，参考 persona）> 有用、清晰。
<persona>
{persona}
</persona>
问题：{item['question']}
{ref}回答1：{first}
回答2：{second}
输出 JSON：{{"winner": "1" 或 "2" 或 "tie", "reason": "一句话"}}""")
            w = j["winner"]
            winner = "tie" if w == "tie" else ((b if w == "1" else a) if swap else (a if w == "1" else b))
            f.write(json.dumps({"id": iid, "category": item["category"], "mode": mode, "winner": winner, "reason": j.get("reason")}, ensure_ascii=False) + "\n")
            print(iid, winner)


def cmd_report(scores: str, pairs: str = None) -> None:
    rows = load_jsonl(scores)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["mode"], r["category"])].append(r)
    print(f"{'label':<14}{'mode':<6}{'category':<14}{'n':>4}  metric")
    for (label, mode, cat), rs in sorted(groups.items()):
        if cat in ("fact", "unanswerable", "update"):
            m = f"acc={statistics.mean(r['score'] for r in rs):.3f}"
        elif cat == "style":
            m = f"persona={statistics.mean(r['persona'] for r in rs):.2f} quality={statistics.mean(r['quality'] for r in rs):.2f}"
        else:
            m = f"quality={statistics.mean(r['quality'] for r in rs):.2f}"
        lat = [r["latency_s"] for r in rs if r.get("latency_s")]
        art = sum(r["artifacts"] for r in rs)
        print(f"{label:<14}{mode:<6}{cat:<14}{len(rs):>4}  {m}  p50_latency={statistics.median(lat):.1f}s  artifacts={art}")
    if pairs:
        tally = defaultdict(lambda: defaultdict(int))
        for p in load_jsonl(pairs):
            tally[(p["mode"], p["category"])][p["winner"]] += 1
        print("\npairwise:")
        for k, v in sorted(tally.items()):
            print(k, dict(v))


def _exists(path: str) -> bool:
    try:
        open(path).close()
        return True
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    {"score": cmd_score, "compare": cmd_compare, "report": cmd_report}[sys.argv[1]](*sys.argv[2:])
