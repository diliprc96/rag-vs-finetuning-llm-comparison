import logging
from evaluation.scorers import grade_mcq, grade_numeric, grade_explanation

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_mcq():
    print("--- Testing MCQ ---")
    # Correct cases
    assert grade_mcq("The answer is B", "B") == 1.0, "Basic Correct Failed"
    assert grade_mcq("Option A", "A") == 1.0, "Option A Failed"
    # Incorrect cases
    assert grade_mcq("I think it is C", "B") == 0.0, "Basic Incorrect Failed"
    print("MCQ Passed")

def test_numeric():
    print("--- Testing Numeric ---")
    # Correct cases
    assert grade_numeric("The velocity is 10.5 m/s", "10.5") == 1.0, "Exact Match Failed"
    assert grade_numeric("It is about 10.4", "10.5", tolerance=0.05) == 1.0, "Tolerance Match Failed"
    # Incorrect cases
    assert grade_numeric("The velocity is 20 m/s", "10.5") == 0.0, "Incorrect Failed"
    print("Numeric Passed")

def test_explanation():
    print("--- Testing Explanation ---")
    ref = "Energy is conserved in a closed system."
    good = "In a closed system, energy remains constant and cannot be created or destroyed."
    bad = "Energy can be created from nothing."
    
    score_good = grade_explanation(good, ref)
    print(f"Good Answer Score: {score_good}")
    assert score_good['score'] >= 0.75, "Good explanation scored too low"
    
    score_bad = grade_explanation(bad, ref)
    print(f"Bad Answer Score: {score_bad}")
    assert score_bad['score'] <= 0.25, "Bad explanation scored too high"
    print("Explanation Passed")

if __name__ == "__main__":
    test_mcq()
    test_numeric()
    test_explanation()
