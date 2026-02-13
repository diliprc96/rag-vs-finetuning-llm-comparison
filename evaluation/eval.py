import scorers


Question_1 = "The rate of change of momentum is equal to: (A) Impulse (B) Force (C) Energy (D) Power"

reference_answer = "The rate of change of an object is equal to the product of the force the object exerts on a secondhand and the time over which the force exerted by one hand on the other."
# ,B,0.0,0.0,0.0,
gradeMCQ = scorers.grade_explanation(Question_1, reference_answer)

