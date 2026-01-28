import re
import os
import json
from anthropic import Anthropic

def get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: No ANTHROPIC_API_KEY set.")
        return None
    return Anthropic(api_key=api_key)

def llm_extract(text, instruction, model="claude-3-haiku-20240307"):
    client = get_client()
    if not client: return None
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{
                "role": "user", 
                "content": f"Extract the requested information from the text. Return ONLY the value, nothing else.\n\nText: {text}\n\nInstruction: {instruction}"
            }]
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"LLM Extraction Error: {e}")
        return None

def grade_mcq(predicted_text, correct_answer):
    """
    Checks if the predicted text matches the correct option (A, B, C, D).
    Robust: Regex -> LLM Fallback (Haiku)
    """
    predicted_text = predicted_text.strip()
    correct_answer = correct_answer.strip().upper()
    
    # 1. Regex Heuristic
    match = re.search(r'\b([A-D])\b', predicted_text.upper())
    if match:
        # Check if the match is plausible (e.g., "The answer is A")
        # Sometimes it matches "A" in "A car is moving..."
        # If the text is short, regex is fine. If long, maybe risky.
        # But let's try strict regex first.
        pred_letter = match.group(1)
        if pred_letter == correct_answer:
            return 1.0
            
    # 2. LLM Fallback (Robustness)
    # If regex failed (or to double check), ask Haiku to extract the option.
    extracted = llm_extract(predicted_text, "Extract the final Multiple Choice Option (A, B, C, or D). Return only the letter.")
    if extracted:
        match_llm = re.search(r'\b([A-D])\b', extracted.upper())
        if match_llm and match_llm.group(1) == correct_answer:
            return 1.0

    return 0.0

def grade_numeric(predicted_text, correct_value, tolerance=0.05):
    """
    Extracts number and compares with tolerance (Increased to 5%).
    Robust: Regex (Last number) -> LLM Fallback
    """
    # 1. Regex Heuristic (Last number)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", predicted_text)
    correct = float(correct_value)
    
    if numbers:
        try:
            val = float(numbers[-1])
            if abs(val - correct) <= tolerance * abs(correct):
                return 1.0
        except:
            pass

    # 2. LLM Fallback
    extracted = llm_extract(predicted_text, "Extract the final numeric answer value. Return only the number.")
    if extracted:
        try:
            # Clean up extracted string
            val_str = re.findall(r"[-+]?\d*\.\d+|\d+", extracted)
            if val_str:
                val_llm = float(val_str[0])
                if abs(val_llm - correct) <= tolerance * abs(correct):
                    return 1.0
        except:
            pass
            
    return 0.0

def grade_explanation(predicted_text, reference_text, rubric=None, client=None):
    """
    Uses Claude to grade the explanation based on a 0-5 scale.
    Upgraded to Claude-3.5-Sonnet for better reasoning.
    """
    if not client:
        client = get_client()
        if not client: return 0.0
        
    model = "claude-sonnet-4-5-20250929" # Updated to available model
    
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
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text
        
        # Robust Extraction: Regex first (ignores bad JSON formatting like unescaped LaTeX)
        match = re.search(r'"score":\s*([0-9.]+)', response)
        if match:
            return float(match.group(1))
            
        # Fallback: Try strict JSON parse
        response_clean = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_clean)
        return float(data.get("score", 0.0))
        
    except Exception as e:
        print(f"Grading error with {model}: {e}")
        # Try fallback model if Sonnet fails (e.g., API error) or extraction fails
        print("Falling back to Haiku...")
        try:
             # Fallback to Haiku
            message = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            response = message.content[0].text
            
            # Regex for Haiku too
            match = re.search(r'"score":\s*([0-9.]+)', response)
            if match:
                return float(match.group(1))
                
            response_clean = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response_clean)
            return float(data.get("score", 0.0))
        except Exception as e2:
             print(f"Fallback Grading error: {e2}")
             return 0.0
