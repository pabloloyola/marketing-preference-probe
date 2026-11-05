"""Small, explicit OpenRouter HTTP adapters; no external runtime dependencies."""
import json
import os
import time
import urllib.error
import urllib.request

URL = "https://openrouter.ai/api"
MODELS = {
    "jev": ("typesafe/jev-1.14", "decisions"),
    "luna_decisions": ("openai/gpt-6-luna-decisions", "decisions"),
    "luna_chat": ("openai/gpt-6-luna", "chat"),
}

class APIError(RuntimeError):
    pass


def request(endpoint, payload, timeout=45):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise APIError("OPENROUTER_API_KEY not set; no API request sent")
    req = urllib.request.Request(
        URL + endpoint,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "Marketing Preference Probe",
        },
        method="POST",
    )
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read(500).decode("utf-8", "replace")
        raise APIError(f"HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise APIError(f"Network error: {e}") from e
    return data, time.perf_counter() - t


def predict(model_key, headline_a, headline_b):
    """Return probability of selecting A as the better *predicted CTR* headline.

    IMPORTANT: model probabilities are NOT claimed to be human click probabilities.
    """
    model, family = MODELS[model_key]
    instruction = (
        "Predict which headline will have the higher click-through rate in a randomized "
        "A/B test on a US news publisher's website. Both alternatives receive comparable "
        "placement and audience exposure. The historical human results are not supplied. "
        "Choose the winner based only on headline wording; do not invent experiment data."
    )
    state = {"context": "Historical US viral-news homepage A/B test", "headline_A": headline_a, "headline_B": headline_b}
    if family == "decisions":
        payload = {"model": model, "state": state, "questions": {"winner": {
            "type": "choice", "instructions": instruction,
            "criteria": {"A": "Headline A has higher click-through rate", "B": "Headline B has higher click-through rate"}
        }}}
        raw, seconds = request("/alpha/decisions", payload)
        answer = raw.get("answers", {}).get("winner", {})
        probs = answer.get("probabilities", {})
        if not isinstance(probs, dict) or "A" not in probs:
            raise APIError(f"Unexpected decision response shape: {str(raw)[:500]}")
        p = float(probs["A"])
    else:
        payload = {"model": model, "temperature": 0, "max_tokens": 1024,
                   "messages": [
                      {"role": "system", "content": instruction + " Respond with ONLY a JSON object with a single field p_A between 0 and 1, indicating your subjective probability that A has higher CTR. This is not a predicted human click share."},
                      {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]}
        raw, seconds = request("/v1/chat/completions", payload)
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            finish_reason = raw.get("choices", [{}])[0].get("finish_reason")
            completion_tokens = (raw.get("usage") or {}).get("completion_tokens")
            raise APIError(
                f"Generative model returned no text (finish_reason={finish_reason!r}, "
                f"completion_tokens={completion_tokens!r}). "
                "This can happen when reasoning consumes the output budget. "
                "Increase max_tokens or investigate the provider response."
            )
        try:
            cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            p = float(json.loads(cleaned)["p_A"])
        except (ValueError, KeyError, TypeError, IndexError) as e:
            raise APIError(f"Could not parse generative prediction: {content[:400]!r}") from e
    if not 0 <= p <= 1:
        raise APIError(f"Invalid probability: {p}")
    usage = raw.get("usage") or {}
    return {"model": model, "model_key": model_key, "p_A": p,
            "winner": "A" if p >= 0.5 else "B", "latency_s": round(seconds, 4),
            "reported_cost_usd": usage.get("cost"), "usage": usage,
            "provider": raw.get("provider"), "served_model": raw.get("model"),
            "response_id": raw.get("id")}
