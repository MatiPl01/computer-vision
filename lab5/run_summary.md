# ST-GCN Run Summary

## Small Subset (2 clips per class)

- **Command:** `python main.py --epochs 5 --batch-size 8 --max-train-per-class 2 --max-val-per-class 2`
- **Log:** `run_small.txt`
- **Highlights:**
  - Training accuracy quickly reaches 100 %, indicating memorisation.
  - Validation accuracy stays at 33 % with predictions collapsing to `act_01`.
  - Confusion matrix confirms all validation clips are classified as the same action.

## Medium Subset (4 clips per class)

- **Command:** `python main.py --epochs 5 --batch-size 4 --max-train-per-class 4 --max-val-per-class 4`
- **Log:** `run_medium.txt`
- **Highlights:**
  - Training accuracy oscillates between 0.4 and 0.75; validation remains at 33 %.
  - The network consistently predicts `act_03`, showing that the dataset is still too small for separation.
  - Precision/recall for `act_01` and `act_02` remain at 0.00.

## Larger Subset (15 clips per class)

- **Command:** `python main.py --epochs 5 --batch-size 8 --max-train-per-class 15 --max-val-per-class 15`
- **Log:** `run_large.txt`
- **Highlights:**
  - Validation accuracy climbs to 67 % by epoch 3 and remains there.
  - `act_01` is perfectly classified; `act_03` reaches recall 1.0 but mixes with `act_02`.
  - `act_02` lacks distinctive cues, yielding 0.0 precision/recall despite more data.

## Full Dataset (all 150 clips per split)

- **Command:** `python main.py --epochs 5 --batch-size 8`
- **Log:** `run_full.txt`
- **Highlights:**
  - Cached loading and the lightweight ST-GCN let the CPU training finish in a few minutes.
  - Validation accuracy steadily improves (0.33 → 0.67 → 1.00 by epoch 5).
  - Final confusion matrix is perfectly diagonal, demonstrating that the model can separate the three actions when enough data is provided.

## Conclusions

- The action classes are not linearly separable with tiny subsets; the model memorises or defaults to a single label.
- Increasing coverage to 15 clips per class improves generalisation, but one class remains confusing, likely due to overlapping motion patterns.
- A full-dataset run resolves the ambiguity entirely, hitting 100 % validation accuracy and confirming the pipeline’s correctness once sufficient data is available.
