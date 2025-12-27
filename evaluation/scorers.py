import re
import os
from anthropic import Anthropic

def grade_mcq(predicted_text, correct_answer):
    """
    Checks if the predicted text matches the correct option (A, B, C, D).
    Simple heuristic: look for the letter or the exact text.
    """
    predicted_text = predicted_text.strip().upper()
    correct_answer = correct_answer.strip().upper()
    
    # If predicted is just "A" or "A)" or "Option A"
    match = re.search(r'\b([A-D])\b', predicted_text)
    if match:
        pred_letter = match.group(1)
        return 1.0 if pred_letter == correct_answer else 0.0
    
    return 0.0

def grade_numeric(predicted_text, correct_value, tolerance=0.01):
    """
    Extracts the last number from predicted text and compares with tolerance.
    """
    # Find all numbers
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", predicted_text)
    if not numbers:
        return 0.0
    
    # Assume the answer is the last number mentioned (heuristic)
    try:
        val = float(numbers[-1])
        correct = float(correct_value)
        if abs(val - correct) <= tolerance * abs(correct):
            return 1.0
        return 0.0
    except:
        return 0.0

def grade_explanation(predicted_text, reference_text, rubric=None, client=None):
    """
    Uses Claude to grade the explanation based on a 0-5 scale.
    """
    if not client:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            client = Anthropic(api_key=api_key)
        else:
            print("Warning: No ANTHROPIC_API_KEY for grading explanation.")
            return 0.0
        
    model = "claude-3-haiku-20240307"
    
    prompt = f"""You are a physics professor grading a student's answer.
    
    Question Context: A physics problem.
    Reference Answer: {reference_text}
    Student Answer: {predicted_text}
    
    Rubric:
    - Correctness (0-2.5 pts)
    - Completeness (0-1.5 pts)
    - Clarity (0-1.0 pts)
    
    Task: Grade the student answer on a scale of 0 to 5.
    Output ONLY a JSON object with the score and reasoning.
    Format: {{"score": 4.5, "reasoning": "..."}}
    """
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text
        # extract json
        import json
        response = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(response)
        return float(data.get("score", 0.0))
    except Exception as e:
        print(f"Grading error: {e}")
        return 0.0
