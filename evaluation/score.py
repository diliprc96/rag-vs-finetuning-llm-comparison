import re
import math
from typing import Optional, Dict


def score_numeric(pred: str, gold: str, rel_tol: float = 0.01) -> float:
    try:
        p = float(re.sub('[^0-9eE+\-.]', '', str(pred)))
        g = float(re.sub('[^0-9eE+\-.]', '', str(gold)))
    except Exception:
        return 0.0
    if math.isclose(p, g, rel_tol=rel_tol):
        return 1.0
    return 0.0


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s\/\^\-\.]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def score_objective(pred: str, gold: str, gold_numeric: Optional[float] = None) -> float:
    if gold_numeric is not None:
        return score_numeric(pred, str(gold_numeric))
    # string match after normalization
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def rubric_score_local(pred: str, gold: str) -> float:
    # Simple heuristic fallback rubric (0-5) when LLM judge not available.
    npreds = normalize_text(pred)
    ngold = normalize_text(gold)
    score = 0.0
    # correctness (50%) -> up to 2.5
    if any(tok in npreds for tok in ngold.split()[:3]):
        score += 2.0
    # completeness (30%) -> up to 1.5
    if len(npreds.split()) >= len(ngold.split()) * 0.6:
        score += 1.0
    # clarity (20%) -> up to 1.0
    if len(npreds.split()) >= 6:
        score += 1.0
    # normalize to 0-5
    return max(0.0, min(5.0, score))


def llm_judge_score(question: str, prediction: str, reference: str, openai_client=None) -> Dict:
    """If `openai_client` is provided (e.g., openai.ChatCompletion), call the judge model (GPT-4o-mini).
    Otherwise fall back to `rubric_score_local`.

    Returns: {"score": float, "explanation": str}
    """
    if openai_client is None:
        return {"score": rubric_score_local(prediction, reference), "explanation": "fallback heuristic"}

    # Example prompt and call - user must provide configured openai client object
    prompt = f"Score the following physics explanation from 0-5 using rubric: conceptual correctness 50%, completeness 30%, clarity 20%.\nQUESTION: {question}\nREFERENCE: {reference}\nPREDICTION: {prediction}\nReturn JSON: {\"score\": <0-5>, \"notes\": <short> }"
    try:
        resp = openai_client.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        txt = resp["choices"][0]["message"]["content"]
        # Very simple parsing attempt
        m = re.search(r"(\d+(?:\.\d+)?)", txt)
        if m:
            return {"score": float(m.group(1)), "explanation": txt}
        return {"score": rubric_score_local(prediction, reference), "explanation": txt}
    except Exception as e:
        return {"score": rubric_score_local(prediction, reference), "explanation": f"error: {e}"}
