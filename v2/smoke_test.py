"""
One-shot smoke test for the Groq + Llama-3.1-8B integration.

Run from the project root:

    python smoke_test.py

It does NOT touch FAISS, embeddings, or any heavy deps — just the LLM call,
so a failure here is unambiguously a Groq / model-id / api-key problem.

Exit code 0 = LLM round-trip succeeded.
Exit code 1 = something blew up; the error is printed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    load_dotenv(Path(__file__).parent / ".env")

    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    key = os.getenv("GROQ_API_KEY", "")

    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print(f"Key      : {'SET (' + str(len(key)) + ' chars)' if key and not key.startswith('your_') else 'MISSING / placeholder'}")
    print()

    if provider != "groq":
        print(f"WARN: LLM_PROVIDER is '{provider}', not 'groq'. This script only smoke-tests Groq.")
        return 1
    if not key or key.startswith("your_"):
        print("ERROR: GROQ_API_KEY is not set in .env.")
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package missing. Run:  pip install -r requirements.txt")
        return 1

    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise financial QA assistant. Answer in one short sentence."},
                {"role": "user", "content": "What is a 10-K filing?"},
            ],
            max_tokens=120,
        )
    except Exception as e:
        print(f"ERROR ({type(e).__name__}): {e}")
        msg = str(e).lower()
        if "decommissioned" in msg or "not found" in msg or "does not exist" in msg or "404" in msg:
            print()
            print("=> The model id appears to be unavailable. Check")
            print("     https://console.groq.com/docs/models")
            print("   for the current model list and update LLM_MODEL in .env.")
        return 1

    dt_ms = (time.perf_counter() - t0) * 1000
    text = resp.choices[0].message.content
    usage = resp.usage

    print(f"Latency  : {dt_ms:.0f} ms")
    print(f"Tokens   : prompt={usage.prompt_tokens}  completion={usage.completion_tokens}")
    print(f"Answer   : {text}")
    print()
    print(f"OK - Groq + {model} is reachable. Safe to proceed to data fetch + indexing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
