# Task: Analyze RAG vs Fine-tuning Results

# Task: Improved Evaluation & Multi-Stage Training

- [ ] Plan and Design <!-- id: 0 -->
    - [ ] Create Implementation Plan <!-- id: 0a -->
    - [ ] User Approval <!-- id: 0b -->
- [ ] Upgrade Evaluation System <!-- id: 1 -->
    - [ ] Modify `scorers.py` to use Claude for all metrics <!-- id: 1a -->
    - [ ] Implement grading scales (0/1 for MCQ/Num, 0-1 for Expl) <!-- id: 1b -->
    - [ ] Define aggregation logic <!-- id: 1c -->
- [ ] Baseline Evaluation <!-- id: 2 -->
    - [ ] Download existing models (Base, Finetuned) <!-- id: 2a -->
    - [ ] Run new evaluation metrics <!-- id: 2b -->
- [x] Data Quality Improvement <!-- id: 4 -->
    - [x] Analyze `alpaca_physics_5k_cleaned.jsonl` inconsistencies <!-- id: 4a -->
    - [x] Report findings vs Proposed Fix <!-- id: 4b -->
    - [x] Implement cleaning logic (Convert 8% JSON to Text) <!-- id: 4c -->
- [x] Phase 4: Multi-Stage Training <!-- id: 5 -->
    - [x] Modify `train.py` for epoch-based checkpoints <!-- id: 5a -->
    - [x] Train for 5 Epochs (Clean Data) <!-- id: 5b -->
    - [x] Evaluate Epoch 3 Checkpoint <!-- id: 5c -->
    - [x] Evaluate Epoch 5 Checkpoint <!-- id: 5d -->
- [x] Final Actions <!-- id: 6 -->
    - [x] Create Prompts Documentation <!-- id: 6a -->
    - [x] Update scorers for relaxed grading <!-- id: 6b -->
    - [x] Upload Model to Hugging Face <!-- id: 6c -->
    - [x] Run "Relaxed" Evaluation <!-- id: 6d -->
- [ ] Final Analysis <!-- id: 5 -->
    - [ ] Compare with Base/RAG <!-- id: 5b -->
    - [ ] Generate Report <!-- id: 5c -->
