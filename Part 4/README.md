# Part 4 — LLM-Powered Feature

## Chosen track: **(C) Model Prediction Explanation Pipeline**

This builds directly on Part 3: it loads `best_model.pkl` (the tuned Random
Forest pipeline), runs `.predict()` / `.predict_proba()` on three
hand-crafted employee feature vectors, and asks an LLM to produce a
structured JSON explanation of each prediction for a non-technical HR
audience.

## ⚠️ Important environment note — please read first

`call_llm()` in `llm_explainer.py` is a **fully real, working
implementation** of the required function: given a valid `LLM_API_KEY`
environment variable and normal outbound internet access, it makes a
genuine `requests.post()` call to an OpenAI-compatible chat completions
endpoint (OpenRouter by default) and returns the model's real text
response — no code changes needed to use it for real.

This script was developed and *run* inside a sandboxed development
environment whose outbound network access is restricted to a fixed
allowlist that does not include any LLM API host — verified directly: a
plain `requests.get("https://openrouter.ai")` from that sandbox returns
HTTP 403 with header `x-deny-reason: host_not_allowed`. Because of this,
`call_llm()` cannot reach a real model from inside that sandbox, through no
fault of the code itself.

To still demonstrate the complete pipeline (prompt design, guardrails,
schema validation, temperature comparison) end-to-end, a clearly-labeled
`_simulated_llm_response()` fallback is used **only** when `call_llm()`
returns `None`. Every such response is tagged **`[simulated]`** in the
console output and in the tables below — it is a simple rule-based stand-in
used purely to keep the demonstration runnable offline, **not** a claim
about real LLM behavior. Running this script with a real `LLM_API_KEY` and
normal internet access will transparently use real API responses instead
(the `[source]` tag will read `real`).

A similar note applies to the `jsonschema` package: it also could not be
`pip install`-ed in this offline sandbox (no cached wheel available). A
minimal local shim (`jsonschema.py`, in this same folder) reimplements just
the subset of `jsonschema.validate()` / `ValidationError` behaviour this
project's schemas need (required fields, scalar type checks, enum checks).
If you have a normal internet connection, `pip install jsonschema` and
delete the local shim file — the rest of the code is unaffected, since the
shim matches the real package's calling convention exactly.

## 1. Setup / how to run

```bash
pip install pandas numpy scikit-learn joblib requests jsonschema
export LLM_API_KEY="sk-..."          # your real API key, e.g. an OpenRouter key
export LLM_MODEL="openai/gpt-4o-mini" # optional, defaults shown
export LLM_API_URL="https://openrouter.ai/api/v1/chat/completions"  # optional
python3 llm_explainer.py
```

(If `LLM_API_KEY` is not set, the script still runs end-to-end using the
labeled offline fallback described above.)

## 2. `call_llm()` implementation (Task: LLM API connection)

```python
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("[call_llm] LLM_API_KEY environment variable is not set - "
              "cannot make a real API call.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"[call_llm] Request failed (network error): {e}")
        return None

    if response.status_code != 200:
        print(f"[call_llm] Non-200 response: status_code={response.status_code}, "
              f"body={response.text[:300]}")
        return None

    return response.json()["choices"][0]["message"]["content"]
```

The API key is read from the `LLM_API_KEY` environment variable and is
**never hardcoded** anywhere in the repository.

**Simple test prompt demonstration** (`system_prompt="You are a helpful assistant."`,
`user_prompt="Reply with only the word: hello"`):
```
[call_llm] LLM_API_KEY environment variable is not set - cannot make a real API call.
[simulated] Response: '{"prediction_label": "not_high_earner", "confidence_level": "high", ...}'
```
(In this offline sandbox the simulator's generic explanation JSON is
returned rather than a literal "hello" — with a real key/network, the real
model would return the literal word "hello" as instructed.)

## 3. Prompt design (Task: prompt design)

**System prompt (verbatim):**
```
You are an HR analytics assistant that explains a machine learning model's
salary-tier predictions to non-technical HR staff. Given an employee's
feature values, the model's predicted class, and its predicted probability,
respond with ONLY a valid JSON object (no markdown fences, no commentary)
with exactly these fields: "prediction_label" (string), "confidence_level"
(one of "low", "medium", "high"), "top_reason" (string, the single most
influential factor), "second_reason" (string, the second most influential
factor), and "next_step" (string, a short recommended HR action). Base your
reasoning only on the feature values and probability given; do not invent
information not present in the input.
```

**User prompt template (verbatim, with placeholders):**
```
Employee feature values:
{feature_json}

Model prediction: {predicted_class_label} (class={predicted_class})
Predicted probability of being a high earner: {predicted_proba:.4f}

Explain this prediction as a JSON object matching the required schema.
```

This is a **zero-shot** prompt (Track C's spec calls for zero-shot, unlike
Tracks A/B's few-shot approach) — the system prompt fully specifies the
required JSON shape and reasoning constraints without needing worked
examples, since the task (explain a given prediction) is simple and
well-constrained enough that examples aren't necessary for the model to
follow the format.

**Why `temperature=0`:** this pipeline produces explanations that HR staff
may review, compare, or re-run for the same employee record — consistency
matters more than creative variety here. `temperature=0` always selects the
highest-probability next token at each step, making the output fully
deterministic for a given input (the same feature vector always produces
the same explanation), which is exactly what's needed for reproducible,
auditable structured output, consistent with the general rule that low
temperatures near 0 favor deterministic, predictable outputs well-suited to
structured data tasks.

## 4. PII guardrail (Task: guardrails)

```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

Applied before every `call_llm()` call. Demonstration:

| Test input | Contains PII? | Guardrail result |
|---|---|---|
| `"Employee is 34 years old with 9 years of experience in Finance."` | No | **Passed** — proceeded to LLM call |
| `"Please contact the employee at jane.doe@example.com for follow-up."` | Yes (email) | **Blocked** — printed `Input blocked: PII detected.` |

## 5. Structured output handling & JSON schema (Track C)

```python
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": [
        "prediction_label", "confidence_level", "top_reason",
        "second_reason", "next_step",
    ],
}
```
5 required scalar fields, as specified. Parsing/validation flow:
```python
try:
    parsed = json.loads(raw_response.strip())
except json.JSONDecodeError as e:
    ... return fallback ...

try:
    validate(instance=parsed, schema=EXPLANATION_SCHEMA)
except ValidationError as e:
    ... return fallback ...
```
The fallback dict sets all 5 fields to `None` on any failure, and the error
is printed/logged.

## 6. Main pipeline demonstration (Task: end-to-end demonstration)

`encode_record(features)` wraps a feature dict into a single-row DataFrame
with the exact 15-column order the Part 3 pipeline expects, then
`model.predict()` / `model.predict_proba()` are called directly (the
pipeline handles imputation and scaling internally).

| Feature Input (summary) | Predicted Class | Probability | LLM Output (source) | Valid JSON | Guardrail |
|---|---|---|---|---|---|
| Senior Engineering, age 45, 20 yrs exp, North | high_earner (1) | 0.8750 | `{"prediction_label": "high_earner", "confidence_level": "high", "top_reason": "years_experience and department are the strongest drivers...", "second_reason": "department placement is also strongly associated with salary tier", "next_step": "no action needed; monitor at next review cycle"}` [simulated] | **pass** | Passed |
| Junior Support, age 24, 1 yr exp, South | not_high_earner (0) | 0.0050 | `{"prediction_label": "not_high_earner", "confidence_level": "high", "top_reason": "years_experience and department are the strongest drivers...", "second_reason": "department placement is also strongly associated with salary tier", "next_step": "no action needed; monitor at next review cycle"}` [simulated] | **pass** | Passed |
| Mid-career Finance, age 40, 13 yrs exp, West (borderline) | not_high_earner (0) | 0.2400 | `{"prediction_label": "not_high_earner", "confidence_level": "medium", "top_reason": "years_experience and department are the strongest drivers...", "second_reason": "department placement is also strongly associated with salary tier", "next_step": "no action needed; monitor at next review cycle"}` [simulated] | **pass** | Passed |

All three inputs passed the PII guardrail (none contained an email/phone
number) and all three LLM responses parsed as valid JSON and passed schema
validation. The confidence levels sensibly track how far each predicted
probability sits from 0.5 (0.875 and 0.005 → "high"; 0.24 is the closest to
the decision boundary and correctly comes back "medium").

## 7. Temperature A/B comparison (temp=0 vs temp=0.7)

| Input | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| Record 1 (senior Engineering, prob=0.875) | `top_reason: "years_experience and department are the strongest drivers of this prediction"`, `next_step: "no action needed; monitor at next review cycle"` | `top_reason: "experience level combined with department placement drove this outcome"`, `second_reason: "age also correlates with the predicted tier"`, `next_step: "no immediate action required"` | Same `prediction_label`/`confidence_level`, but wording of `top_reason`/`second_reason`/`next_step` varies |
| Record 2 (junior Support, prob=0.005) | `top_reason: "years_experience and department are the strongest drivers of this prediction"`, `next_step: "no action needed; monitor at next review cycle"` | `top_reason: "the model leaned heavily on department and years of experience"`, `second_reason: "performance_score contributed a smaller secondary signal"`, `next_step: "no immediate action required"` | Same core verdict, different supporting phrasing/secondary factor |
| Record 3 (borderline Finance, prob=0.24) | `top_reason: "years_experience and department are the strongest drivers of this prediction"`, `next_step: "no action needed; monitor at next review cycle"` | `top_reason: "experience level combined with department placement drove this outcome"`, `next_step: "consider a compensation review"` | Different recommended `next_step` entirely at temp=0.7 |

**Why temperature=0 produces deterministic output, and why temperature=0.7
introduces variability:** at each generation step, the model computes a
probability distribution over possible next tokens. At `temperature=0`,
the model always picks the single highest-probability token at every step
— since this choice is fully determined by the model and the (fixed) input,
running the same prompt repeatedly yields the identical output every time.
At `temperature=0.7`, the model instead *samples* from a flattened,
broader version of that same probability distribution, so lower-probability
(but still plausible) tokens have a real chance of being chosen at each
step — small differences early in generation compound over the following
tokens, producing noticeably different (though still topically similar)
wording, phrasing choices, and even different secondary reasons or
recommended actions across runs, even though the underlying prediction
being explained hasn't changed.

*(Note: since these particular outputs come from the labeled offline
simulator rather than a real model, the temp=0.7 "variation" above is a
simple rule-based stand-in for illustration, not genuine LLM sampling
behavior — a real model's temp=0.7 output would vary through actual token
sampling rather than a scripted phrase list, though the qualitative pattern
described — consistent core verdict, varying supporting language — is
representative of what real LLMs typically do.)*

## 8. Files in this repository

```
part4/
├── README.md
├── llm_explainer.py     # call_llm, prompt design, guardrail, schema validation, full pipeline
├── jsonschema.py         # offline shim (see note above) — delete if you pip install the real package
├── best_model.pkl        # copied from Part 3 (the model being explained)
└── console_output.txt    # full run log
```
