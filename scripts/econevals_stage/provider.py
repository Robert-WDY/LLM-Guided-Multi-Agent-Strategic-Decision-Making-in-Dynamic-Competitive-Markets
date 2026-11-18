"""Bound real requests; ledger is written before transmission, including failed attempts."""
import json
import os
import re
import time
from openai import OpenAI
from common import RAW

MODELS = {"doubao": "doubao-seed-2-0-lite-260428", "deepseek": "deepseek-v4-flash"}
URLS = {"doubao": "https://ark.cn-beijing.volces.com/api/v3", "deepseek": "https://api.deepseek.com"}
KEYS = {"doubao": ["ARK_API_KEY","ARK_FALLBACK_KEY"], "deepseek":["DEEPSEEK_API_KEY"]}


def record(path, row):
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def decision(provider, identifier, prompt, lo, hi):
    # Each job has one writer per provider/phase; caps partition the global 500 calls.
    phase = identifier.split(":")[0]
    ledger = RAW / f"{provider}-{phase}-calls.jsonl"
    log = [json.loads(s) for s in ledger.read_text(encoding="utf8").splitlines()] if ledger.exists() else []
    for event in reversed(log):
        if event["id"] == identifier and "price" in event:
            return event["price"]
    caps = {"basic": 120, "window": 64, "official": 124, "pilot": 65}
    cap = caps[phase]
    # The stage-level cap counts reservations even if a response was lost.
    def sent_total():
        return sum(1 for f in RAW.glob("*-calls.jsonl") for line in f.read_text(encoding="utf8").splitlines()
                   if json.loads(line)["event"] == "sent")
    if len(prompt.encode()) > 16000:
        raise ValueError("prompt exceeded stage input cap")
    for attempt in range(3):
        n = sum(e["event"] == "sent" for e in log)
        if n >= cap or sent_total() >= 500:
            raise RuntimeError("bounded call cap")
        item = {"event":"sent","id":identifier,"attempt":n+1,"model":MODELS[provider]}
        record(ledger,item)
        log.append(item)
        key_name = KEYS[provider][min(attempt,len(KEYS[provider])-1)]
        client = OpenAI(base_url=URLS[provider],api_key=os.environ[key_name],timeout=23,max_retries=0)
        try:
            options = {}
            if phase == "official":
                options["tools"] = [{"type":"function","function":{"name":"set_prices","description":"Submit the final numeric price for Product_1.","parameters":{"type":"object","properties":{"price":{"type":"number"}},"required":["price"]}}}]
                options["tool_choice"] = {"type":"function","function":{"name":"set_prices"}}
            result = client.chat.completions.create(model=MODELS[provider],temperature=0,
                messages=[{"role":"user","content":prompt}],max_tokens=96,**options,extra_body={"thinking":{"type":"disabled"}})
            content = result.choices[0].message.content or ""
            calls = result.choices[0].message.tool_calls or []
            if calls and phase == "official":
                content = calls[0].function.arguments
            usage = result.usage.model_dump() if result.usage else {}
            parsed = re.fullmatch(r"\s*(?:\x60{3}(?:json)?\s*)?(?:\{\s*\"price\"\s*:\s*)?(-?\d+(?:\.\d+)?)(?:\s*\})?(?:\s*\x60{3})?\s*",content)
            price = float(parsed[1]) if parsed else None
            out = {"event":"response","id":identifier,"usage":usage,"valid":price is not None and lo <= price <= hi}
            if out["valid"]:
                out["price"] = price
            record(ledger,out)
            if out["valid"]:
                return price
            prompt += " Reply with only one number within the stated bounds."
        except Exception as exc:
            status = getattr(exc,"status_code",None)
            record(ledger,{"event":"error","id":identifier,"type":type(exc).__name__,"status":status,"code":str(getattr(exc,"code",None))[:80]})
            if status in (401,403,402,404) and attempt+1 >= len(KEYS[provider]):
                raise RuntimeError(f"{provider} unavailable ({status})") from None
            time.sleep(0.3)
    raise RuntimeError(f"{provider}: no valid decision after bounded retries")

