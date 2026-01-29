import os
import re
import json
import logging
import math
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_claude_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("Error: ANTHROPIC_API_KEY not found in environment variables.")
        return None
    return Anthropic(api_key=api_key)

def grade_mcq(predicted_text: str, reference_answer: str, client: Optional[Anthropic] = None) -> float:
    """
    Grades MCQ using Claude to extract the answer and compare.
    Returns 1.0 if correct, 0.0 otherwise.
    """
    if not client:
        client = get_claude_client()
        if not client:
            return 0.0

    prompt = f"""
    You are an impartial grader.
    Task: Identify the selected option (A, B, C, or D) from the Student's Answer and compare it to the Correct Answer.
    
    Student's Answer: "{predicted_text}"
    Correct Answer: "{reference_answer}"
    
    Instructions:
    1. If the Student's Answer matches the Correct Answer (A, B, C, or D) or clearly states the text of the correct option, score 1.
    2. If the answer is correct in concept but verbose (e.g., writes out the full sentence instead of just 'A'), score 1.
    3. Ignore formatting artifacts or extra text.
    4. Only score 0 if the answer is fundamentally wrong or selects a different option.
    5. Output ONLY a valid JSON object with the format: {{"score": 0}} or {{"score": 1}}. Do not output any other text.
    """

    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text.strip()
        
        # Extract JSON
        match = re.search(r'\{.*"score":\s*([0-1]).*\}', response_text, re.DOTALL)
        if match:
             # simple regex for 0 or 1
            return float(match.group(1))
        
        # Try full JSON parse if regex fails
        data = json.loads(response_text)
        return float(data.get("score", 0))

    except Exception as e:
        logger.error(f"Error in grade_mcq: {e}")
        return 0.0

def grade_numeric(predicted_text: str, reference_answer: str, tolerance: float = 0.05, client: Optional[Anthropic] = None) -> float:
    """
    Grades numeric answers using Claude to extract the value and compare.
    Returns 1.0 if within tolerance, 0.0 otherwise.
    """
    if not client:
        client = get_claude_client()
        if not client:
            return 0.0

    prompt = f"""
    You are an impartial grader.
    Task: Extract the numeric value from the Student's Answer and determine if it matches the Correct Answer.
    
    Student's Answer: "{predicted_text}"
    Correct Answer: "{reference_answer}"
    Tolerance: +/- {tolerance*100}%
    
    Instructions:
    1. Identify the final numeric answer in the Student's Answer. Extract it even if buried in text or JSON.
    2. Ignore units unless they are fundamentally wrong (e.g. asking for meters but given seconds).
    3. Compare it to the Correct Answer value.
    4. If the value is within {tolerance*100}% of the Correct Answer, score 1. Otherwise, score 0.
    5. Output ONLY a valid JSON object with the format: {{"score": 0}} or {{"score": 1}}. Do not output any other text.
    """

    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text.strip()
        
        # Extract JSON
        match = re.search(r'\{.*"score":\s*([0-1]).*\}', response_text, re.DOTALL)
        if match:
            return float(match.group(1))
            
        data = json.loads(response_text)
        return float(data.get("score", 0))

    except Exception as e:
        logger.error(f"Error in grade_numeric: {e}")
        return 0.0


def grade_explanation(predicted_text: str, reference_text: str, rubric=None, client: Optional[Anthropic] = None) -> Dict[str, Any]:
    """
    Grades explanations using Claude on a 5-point scale (0, 0.25, 0.5, 0.75, 1.0).
    Returns a dict with 'score' and 'reasoning'.
    """
    if not client:
        client = get_claude_client()
        if not client:
            return {"score": 0.0, "reasoning": "API Key missing"}

    # Use Sonnet for better reasoning on explanations
    model = "claude-sonnet-4-5-20250929" 

    prompt = f"""
    You are an expert Physics grader.
    Task: Grade the Student's Explanation against the Reference Explanation.
    
    Student's Answer: "{predicted_text}"
    Reference Answer: "{reference_text}"
    
    Rubric:
    - 1.0: Correct. Captures the core physical concept. Ignore formatting, length, or extra "chatty" text.
    - 0.75: Good. Correct core concept but misses minor details.
    - 0.5: Weak. Captures some correct keywords but misses the main logic.
    - 0.25: Poor. Barely relevant or mostly incorrect.
    - 0.0: Wrong. Completely incorrect, irrelevant, or IDK.
    
    Instructions:
    1. Compare the core physical meaning. Do not penalize for verbosity or formatting.
    2. Focus on whether the student understands the physics.
    3. Output ONLY a valid JSON object with: {{"score": <float>, "reasoning": "<short text>"}}.
    4. Allowed scores: [0.0, 0.25, 0.5, 0.75, 1.0].
    """

    try:
        message = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text.strip()
        
        # Clean potential markdown
        response_clean = response_text.replace("```json", "").replace("```", "").strip()
        
        # Try finding json bracket
        start = response_clean.find("{")
        end = response_clean.rfind("}")
        if start != -1 and end != -1:
            response_clean = response_clean[start:end+1]

        data = json.loads(response_clean)
        return {
            "score": float(data.get("score", 0.0)),
            "reasoning": data.get("reasoning", "No reasoning provided")
        }

    except Exception as e:
        logger.error(f"Error in grade_explanation: {e}")
        # Fallback to Haiku if Sonnet fails
        try:
            logger.info("Falling back to Haiku for explanation grading...")
            message = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=300,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            response_clean = message.content[0].text.strip().replace("```json", "").replace("```", "").strip()
            start = response_clean.find("{")
            end = response_clean.rfind("}")
            if start != -1 and end != -1:
                response_clean = response_clean[start:end+1]
                
            data = json.loads(response_clean)
            return {
                "score": float(data.get("score", 0.0)),
                "reasoning": data.get("reasoning", "Fallback grading")
            }
        except Exception as e2:
            logger.error(f"Fallback failed: {e2}")
            return {"score": 0.0, "reasoning": f"Error: {str(e)}"}
