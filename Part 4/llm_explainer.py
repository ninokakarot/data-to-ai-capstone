"""
Part 4 - Track C: Model Prediction Explanation Pipeline

Loads the best model from Part 3 (best_model.pkl), runs .predict() and
.predict_proba() on three hand-crafted feature vectors, then asks an LLM to
produce a structured JSON explanation for each prediction - validated
against a JSON schema, with a PII guardrail applied before every LLM call.

--------------------------------------------------------------------------
IMPORTANT - NETWORK ACCESS NOTE (read this first)
--------------------------------------------------------------------------
call_llm() below is a fully real, working implementation: given a valid
LLM_API_KEY environment variable and outbound internet access, it will
make a genuine HTTP POST to an OpenAI-compatible chat completions endpoint
(default: OpenRouter) and return the model's real text response.

This script was developed and *run* inside a sandboxed environment whose
outbound network access is restricted to an allowlist that does not
include any LLM API host (verified: requests to openrouter.ai return
HTTP 403 "host_not_allowed" from the sandbox's egress proxy). Because of
this, call_llm() cannot reach a real model from inside that sandbox.

To still demonstrate the full pipeline end-to-end (schema validation,
guardrails, prompt design, temperature comparison), a clearly-labeled
`_simulated_llm_response()` fallback is used ONLY when call_llm() returns
None. Every simulated response is printed and reported with an explicit
"[SIMULATED - NO NETWORK ACCESS]" tag - it is a rule-based stand-in, not a
real model output, and it is not used anywhere to make a claim about a
real LLM's behaviour. Anyone running this script with a valid LLM_API_KEY
on a machine with normal internet access will get real API responses
instead, with no code changes required.
--------------------------------------------------------------------------

Run with: python3 llm_explainer.py
"""
import os
import re
import json
import random
import joblib
import requests
import numpy as np
import pandas as pd
from jsonschema import validate, ValidationError

pd.set_option("display.width", 120)

# ===========================================================================
# LLM API CONFIGURATION
# ===========================================================================
LLM_API_URL = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    """
    Real call_llm implementation per the task spec:
    - reads the API key from an environment variable (never hardcoded)
    - builds a standard OpenAI-compatible chat completions payload
    - sets Authorization + Content-Type headers
    - POSTs to LLM_API_URL
    - returns None (after printing the status code) on any non-200 response
    - otherwise returns response.json()['choices'][0]['message']['content']
    """
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


# ===========================================================================
# OFFLINE FALLBACK - used ONLY when call_llm() returns None (see notice above)
# ===========================================================================
_TOP_REASON_PHRASINGS_T0 = "years_experience and department are the strongest drivers of this prediction"
_TOP_REASON_PHRASINGS_T7 = [
    "years_experience and department appear to be the biggest factors here",
    "the model leaned heavily on department and years of experience",
    "experience level combined with department placement drove this outcome",
]

def _simulated_llm_response(feature_values, predicted_class, predicted_proba, temperature):
    """
    Rule-based stand-in for a real LLM response, used only because this
    sandbox cannot reach a real LLM API (see module docstring). Produces a
    schema-shaped JSON explanation string. At temperature=0 it is fully
    deterministic; at temperature=0.7 it introduces cosmetic wording
    variation, loosely mimicking (NOT reproducing) how a real LLM's
    sampling temperature affects output variability.
    """
    label = "high_earner" if predicted_class == 1 else "not_high_earner"
    conf = "high" if (predicted_proba > 0.8 or predicted_proba < 0.2) else "medium"

    if temperature == 0.0:
        top_reason = _TOP_REASON_PHRASINGS_T0
        second_reason = "department placement is also strongly associated with salary tier"
        next_step = "no action needed; monitor at next review cycle"
    else:
        rng = random.Random(hash(json.dumps(feature_values, sort_keys=True)) & 0xffff)
        top_reason = rng.choice(_TOP_REASON_PHRASINGS_T7)
        second_reason = rng.choice([
            "age also correlates with the predicted tier",
            "performance_score contributed a smaller secondary signal",
            "bonus_pct nudged the prediction slightly",
        ])
        next_step = rng.choice([
            "consider a compensation review",
            "flag for HR follow-up next cycle",
            "no immediate action required",
        ])

    resp = {
        "prediction_label": label,
        "confidence_level": conf,
        "top_reason": top_reason,
        "second_reason": second_reason,
        "next_step": next_step,
    }
    return json.dumps(resp)


def call_llm_or_simulate(system_prompt, user_prompt, feature_values, predicted_class,
                          predicted_proba, temperature=0.0, max_tokens=512):
    """Tries the real call_llm(); falls back to the labeled simulator if it returns None."""
    real_response = call_llm(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
    if real_response is not None:
        return real_response, "real"
    simulated = _simulated_llm_response(feature_values, predicted_class, predicted_proba, temperature)
    return simulated, "simulated"


# ===========================================================================
# PII GUARDRAIL
# ===========================================================================
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


# ===========================================================================
# JSON SCHEMA (Track C: >= 5 required scalar fields)
# ===========================================================================
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

FALLBACK_EXPLANATION = {
    "prediction_label": None,
    "confidence_level": None,
    "top_reason": None,
    "second_reason": None,
    "next_step": None,
}


def parse_and_validate(raw_response):
    """Strip whitespace -> json.loads (catch JSONDecodeError) -> jsonschema.validate
    (catch ValidationError). Returns (parsed_dict_or_fallback, status_string)."""
    if raw_response is None:
        return dict(FALLBACK_EXPLANATION), "fail (no response)"

    try:
        parsed = json.loads(raw_response.strip())
    except json.JSONDecodeError as e:
        print(f"[parse_and_validate] JSONDecodeError: {e}")
        return dict(FALLBACK_EXPLANATION), f"fail (JSONDecodeError: {e})"

    try:
        validate(instance=parsed, schema=EXPLANATION_SCHEMA)
    except ValidationError as e:
        print(f"[parse_and_validate] ValidationError: {e.message}")
        return dict(FALLBACK_EXPLANATION), f"fail (ValidationError: {e.message})"

    return parsed, "pass"


# ===========================================================================
# TASK: encode_record + model loading
# ===========================================================================
FEATURE_NAMES = [
    "age", "education_level", "remote_work", "years_experience",
    "performance_score", "satisfaction_score", "bonus_pct",
    "department_Finance", "department_HR", "department_Marketing",
    "department_Sales", "department_Support",
    "region_North", "region_South", "region_West",
]

def encode_record(features: dict) -> pd.DataFrame:
    """Turn a feature dict into a single-row DataFrame with the exact column
    order the Part 3 pipeline was trained on."""
    return pd.DataFrame([features])[FEATURE_NAMES]


print("=" * 80)
print("Loading best_model.pkl (from Part 3)")
print("=" * 80)
model = joblib.load("best_model.pkl")
print("Model loaded successfully:", type(model))

# ===========================================================================
# Three hand-crafted feature-vector inputs
# ===========================================================================
test_records = [
    {  # Senior Engineering employee, high experience
        "age": 45.0, "education_level": 3, "remote_work": 0,
        "years_experience": 20.0, "performance_score": 8.0,
        "satisfaction_score": 75.0, "bonus_pct": 5.0,
        "department_Finance": 0, "department_HR": 0, "department_Marketing": 0,
        "department_Sales": 0, "department_Support": 0,
        "region_North": 1, "region_South": 0, "region_West": 0,
    },
    {  # Junior Support employee, low experience
        "age": 24.0, "education_level": 0, "remote_work": 1,
        "years_experience": 1.0, "performance_score": 6.5,
        "satisfaction_score": 60.0, "bonus_pct": 1.0,
        "department_Finance": 0, "department_HR": 0, "department_Marketing": 0,
        "department_Sales": 0, "department_Support": 1,
        "region_North": 0, "region_South": 1, "region_West": 0,
    },
    {  # Mid-career Finance employee, borderline case
        "age": 40.0, "education_level": 2, "remote_work": 0,
        "years_experience": 13.0, "performance_score": 7.5,
        "satisfaction_score": 68.0, "bonus_pct": 3.5,
        "department_Finance": 1, "department_HR": 0, "department_Marketing": 0,
        "department_Sales": 0, "department_Support": 0,
        "region_North": 0, "region_South": 0, "region_West": 1,
    },
]

# ===========================================================================
# Prompt design
# ===========================================================================
SYSTEM_PROMPT = (
    "You are an HR analytics assistant that explains a machine learning "
    "model's salary-tier predictions to non-technical HR staff. Given an "
    "employee's feature values, the model's predicted class, and its "
    "predicted probability, respond with ONLY a valid JSON object (no "
    "markdown fences, no commentary) with exactly these fields: "
    '"prediction_label" (string), "confidence_level" (one of "low", '
    '"medium", "high"), "top_reason" (string, the single most influential '
    'factor), "second_reason" (string, the second most influential factor), '
    'and "next_step" (string, a short recommended HR action). '
    "Base your reasoning only on the feature values and probability given; "
    "do not invent information not present in the input."
)

USER_PROMPT_TEMPLATE = (
    "Employee feature values:\n{feature_json}\n\n"
    "Model prediction: {predicted_class_label} (class={predicted_class})\n"
    "Predicted probability of being a high earner: {predicted_proba:.4f}\n\n"
    "Explain this prediction as a JSON object matching the required schema."
)

print("\n" + "=" * 80)
print("PROMPT DESIGN")
print("=" * 80)
print("\nSYSTEM PROMPT:\n", SYSTEM_PROMPT)
print("\nUSER PROMPT TEMPLATE:\n", USER_PROMPT_TEMPLATE)
print("\nTemperature = 0.0 chosen for the main pipeline: this task requires "
      "consistent, reproducible structured JSON explanations that a human "
      "reviewer can trust to be stable across repeated runs for the same "
      "input - temperature=0 always selects the highest-probability next "
      "token, removing sampling randomness.")

# ===========================================================================
# Simple test prompt demonstration (call_llm sanity check)
# ===========================================================================
print("\n" + "=" * 80)
print("SIMPLE TEST PROMPT DEMONSTRATION")
print("=" * 80)
test_response, test_source = call_llm_or_simulate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Reply with only the word: hello",
    feature_values={}, predicted_class=0, predicted_proba=0.0,
    temperature=0.0,
)
print(f"[{test_source}] Response: {test_response!r}")

# ===========================================================================
# PII GUARDRAIL DEMONSTRATION
# ===========================================================================
print("\n" + "=" * 80)
print("PII GUARDRAIL DEMONSTRATION")
print("=" * 80)

clean_input = "Employee is 34 years old with 9 years of experience in Finance."
pii_input = "Please contact the employee at jane.doe@example.com for follow-up."

for label, text in [("clean_input", clean_input), ("pii_input", pii_input)]:
    if has_pii(text):
        print(f"{label}: BLOCKED - PII detected in: {text!r}")
        print("Input blocked: PII detected.")
    else:
        print(f"{label}: PASSED guardrail - proceeding to LLM call: {text!r}")

# ===========================================================================
# Main pipeline: predict + explain for all 3 records (temperature=0)
# ===========================================================================
print("\n" + "=" * 80)
print("MAIN PIPELINE: predict() + predict_proba() + LLM explanation (temp=0)")
print("=" * 80)

main_results = []
for i, features in enumerate(test_records):
    encoded = encode_record(features)
    pred_class = int(model.predict(encoded)[0])
    pred_proba = float(model.predict_proba(encoded)[0, 1])
    pred_label = "high_earner" if pred_class == 1 else "not_high_earner"

    feature_json = json.dumps(features, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        feature_json=feature_json,
        predicted_class_label=pred_label,
        predicted_class=pred_class,
        predicted_proba=pred_proba,
    )

    # PII guardrail check before every LLM call
    if has_pii(user_prompt):
        print(f"\nRecord {i+1}: Input blocked: PII detected.")
        raw_response, source = None, "blocked"
        parsed, status = dict(FALLBACK_EXPLANATION), "blocked (PII detected)"
    else:
        raw_response, source = call_llm_or_simulate(
            SYSTEM_PROMPT, user_prompt, features, pred_class, pred_proba, temperature=0.0
        )
        parsed, status = parse_and_validate(raw_response)

    print(f"\n--- Record {i+1} ---")
    print("Feature values:", features)
    print("Predicted class:", pred_class, f"({pred_label})")
    print("Predicted probability:", round(pred_proba, 4))
    print(f"LLM raw response [{source}]:", raw_response)
    print("Validation status:", status)
    print("Parsed explanation:", parsed)

    main_results.append({
        "record": i + 1,
        "features": features,
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "predicted_proba": pred_proba,
        "raw_response": raw_response,
        "source": source,
        "validation_status": status,
        "parsed": parsed,
    })

# ===========================================================================
# Temperature A/B comparison (temp=0 vs temp=0.7) for all 3 records
# ===========================================================================
print("\n" + "=" * 80)
print("TEMPERATURE A/B COMPARISON (temp=0 vs temp=0.7)")
print("=" * 80)

temp_comparison_results = []
for i, features in enumerate(test_records):
    encoded = encode_record(features)
    pred_class = int(model.predict(encoded)[0])
    pred_proba = float(model.predict_proba(encoded)[0, 1])
    pred_label = "high_earner" if pred_class == 1 else "not_high_earner"

    feature_json = json.dumps(features, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        feature_json=feature_json,
        predicted_class_label=pred_label,
        predicted_class=pred_class,
        predicted_proba=pred_proba,
    )

    resp_t0, src_t0 = call_llm_or_simulate(
        SYSTEM_PROMPT, user_prompt, features, pred_class, pred_proba, temperature=0.0
    )
    resp_t7, src_t7 = call_llm_or_simulate(
        SYSTEM_PROMPT, user_prompt, features, pred_class, pred_proba, temperature=0.7
    )

    print(f"\n--- Record {i+1} ---")
    print(f"temp=0.0 [{src_t0}]:", resp_t0)
    print(f"temp=0.7 [{src_t7}]:", resp_t7)

    temp_comparison_results.append({
        "record": i + 1, "temp0": resp_t0, "temp7": resp_t7,
        "source0": src_t0, "source7": src_t7,
    })

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
