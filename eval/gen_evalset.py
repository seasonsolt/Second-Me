"""Build a frozen eval set from the documents uploaded to a Second-Me instance.

Usage: python gen_evalset.py <path-to-lpm.db> <out.json>
Uses the Magpie OpenAI-compatible gateway as generator. Run once; the output is
frozen and reused for every version being compared.
"""
import hashlib
import json
import re
import sqlite3
import sys

import os

import requests

GATEWAY = os.getenv("EVAL_GATEWAY", "http://127.0.0.1:3425/v1/chat/completions")
GEN_MODEL = os.getenv("EVAL_GEN_MODEL", "codex/gpt-6.1-sol")
MAX_DOC_CHARS = 24000


def llm_json(prompt: str) -> object:
    r = requests.post(
        GATEWAY,
        json={"model": GEN_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
        timeout=600,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return json.loads(m.group(1) if m else text)


def fact_questions(name: str, content: str, n: int) -> list:
    prompt = f"""你在为一个"个人数字分身"构建事实问答评测集。下面是分身主人上传的一份文档《{name}》。
请出 {n} 道题，要求：
- 以第二人称向分身提问（例如"你在哪一年加入……？"），像真实用户会问的问题；
- 答案必须能在文档中直接找到，短而可核对（一个名称/数字/年份/短语或一句话）；
- 覆盖文档不同部分，不要出需要推理多步或主观判断的题；
- 不要出涉及密码、密钥、身份证号、手机号等敏感信息的题。
只输出 JSON 数组：[{{"question": "...", "answer": "...", "evidence": "文档中的原句（不超过60字）"}}]

文档：
{content[:MAX_DOC_CHARS]}"""
    return llm_json(prompt)


def main(db_path: str, out_path: str) -> None:
    db = sqlite3.connect(db_path)
    docs = db.execute("select id, name, raw_content from document where raw_content is not null").fetchall()
    total = sum(len(c) for _, _, c in docs)

    items = []
    for doc_id, name, content in docs:
        n = max(1, min(8, round(40 * len(content) / total)))
        for q in fact_questions(name, content, n):
            items.append({"category": "fact", "source": name, "doc_id": doc_id, **q})
        print(f"fact: {name} -> {n}", file=sys.stderr)

    titles = "\n".join(f"- {name}" for _, name, _ in docs)
    profile = docs[0][2][:6000]
    unanswerable = llm_json(f"""分身主人上传了以下文档（标题和一份概况节选）。请出 10 个以第二人称提问、听起来合理、
但这些文档里**肯定没有答案**的个人问题（如家人姓名、具体某天的私事、未提及的经历），用于测试分身会不会编造。
只输出 JSON 数组：[{{"question": "..."}}]
文档标题：
{titles}
概况节选：
{profile}""")
    items += [{"category": "unanswerable", **q} for q in unanswerable]

    style = llm_json(f"""根据下面的文档标题和概况节选，出 10 个开放式问题，让分身用主人的口吻回答，
用来评估"说话风格、价值判断、技术观点像不像本人"（例如对某种架构取舍的看法、给新人的建议、怎么做技术决策）。
问题要贴合主人的领域，但答案不能是简单事实。只输出 JSON 数组：[{{"question": "..."}}]
文档标题：
{titles}
概况节选：
{profile}""")
    items += [{"category": "style", **q} for q in style]

    general = [
        "用三句话解释一下 CAP 定理。",
        "帮我把这句话改得更专业：这个接口老是挂，大家都很烦。",
        "Write a short English status update saying the release is delayed by two days due to a failing integration test.",
        "比较一下 Kafka 和 RabbitMQ 各自适合的场景，简洁列点。",
        "写一个 Python 函数，判断一个字符串是否是回文，忽略大小写和空格。",
    ]
    items += [{"category": "general", "question": q} for q in general]

    # Fictional facts used later to test add/modify without retraining.
    items += [
        {"category": "update", "question": "你最近在学什么乐器？", "answer": "大提琴",
         "add_fact": "2026年10月我开始学习大提琴，每周六上午上课。",
         "modified_fact": "2026年10月我开始学习大提琴，后来改学了古典吉他。", "modified_answer": "古典吉他"},
        {"category": "update", "question": "你给家里的猫取了什么名字？", "answer": "芝士",
         "add_fact": "我家里养了一只橘猫，名字叫芝士。",
         "modified_fact": "我家的橘猫原来叫芝士，后来改名叫年糕。", "modified_answer": "年糕"},
    ]

    for i, it in enumerate(items):
        it["id"] = f"{it['category']}-{i:03d}"
    payload = json.dumps(items, ensure_ascii=False, indent=2)
    with open(out_path, "w") as f:
        f.write(payload)
    print(f"{len(items)} items, sha256={hashlib.sha256(payload.encode()).hexdigest()[:12]}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
