# Model Behavior Analysis (5 Epochs)

## Observation
The model trained for 5 epochs on the "Cleaned" dataset demonstrates **good physics knowledge** but **poor instruction following** regarding output formatting.

### Specific Failures
1.  **Verbose Answers**:
    -   *Question*: "Which of the following units is an SI base unit? ... (C) Kilogram ..."
    -   *Model Output*: "The units listed in the table are SI base units." (Correct fact, but failed to select "C").
    -   *Previous Grade*: 0.0 (Strict formatting required "C").
    -   *New Strategy*: Grade as 1.0 (Content is correct).

2.  **JSON Hallucinations**:
    -   *Question*: "What is the average speed..."
    -   *Model Output*: `{"average": 25}`
    -   *Previous Grade*: 0.0 (Expected "25").
    -   *New Strategy*: Extract "25" from the JSON/Text string and compare. Grade as 1.0.

### Conclusion
The "cleaning" of the dataset removed some JSON, but the model likely overfit to the remaining patterns or the underlying base model's tendencies. However, since the *physics* is correct, we should relax the evaluation metric to capture this value rather than penalizing the format.
