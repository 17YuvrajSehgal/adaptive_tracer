# LSTM Results Comparison: Paper 5 Duration Categories

This note compares the three LSTM runs trained on the LMAT microservice dataset with the paper-style **5 duration categories** setting. In the code and output directories this appears as `cats6`, because class `0` is reserved for padding/non-exit positions and the real duration bins are `1..5`.

The comparison below is based on:

- `train_rank0.log` from each run for training behavior and best validation checkpoints
- `ood_results.json` from each run for final OOD detection metrics

Runs compared:

- Event-only: [train_rank0.log](C:\workplace\adaptive_tracer\logs\lstm\lstm-5-event\train_rank0.log)
- Duration-only: [train_rank0.log](C:\workplace\adaptive_tracer\logs\lstm\lstm-5-duration\train_rank0.log)
- Multi-task: [train_rank0.log](C:\workplace\adaptive_tracer\logs\lstm\lstm-5-multitask\train_rank0.log)

## OOD Detection Summary

| Test set | Event AUROC | Event AUPR | Event F1 | Duration AUROC | Duration AUPR | Duration F1 | Multi-task AUROC | Multi-task AUPR | Multi-task F1 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| CPU | 0.6238 | 0.6238 | 0.6687 | 0.6403 | 0.6179 | 0.6706 | 0.6342 | 0.6224 | 0.6606 |
| Disk | 0.5704 | 0.5910 | 0.6667 | 0.6321 | 0.6067 | 0.6569 | 0.6194 | 0.6124 | 0.6494 |
| Memory | 0.6234 | 0.6405 | 0.6762 | 0.7874 | 0.7882 | 0.6678 | 0.7922 | 0.7945 | 0.6667 |
| Network | 0.7316 | 0.7690 | 0.6862 | 0.6994 | 0.6874 | 0.6878 | 0.7556 | 0.7651 | 0.7259 |
| **Average** | **0.6373** | **0.6561** | **0.6744** | **0.6898** | **0.6751** | **0.6708** | **0.7004** | **0.6986** | **0.6756** |

## Training Summary

| Run | Best val loss | Final train accuracy | Final epoch summary | Runtime |
| --- | ---:| --- | --- | --- |
| Event-only | 0.1350 | Event acc `95.40%` | `loss=0.1184`, `acc_e=95.40%` | about 4h 23m |
| Duration-only | 0.7420 | Duration acc `71.30%` | `loss=0.6766`, `acc_l=71.30%` | about 4h 24m |
| Multi-task | 0.4377 | Event acc `94.98%`, Duration acc `70.06%` | `loss=0.4179`, `acc_e=94.98%`, `acc_l=70.06%` | about 4h 30m |

## Short Discussion

The three runs are all stable and converged normally. The event-only model learned next-event prediction very well and reached the strongest standalone event accuracy, but its OOD ranking quality was only moderate overall, especially on disk and memory anomalies.

The duration-only model improved the anomaly-detection signal substantially, especially for **memory stress**, where its AUROC and AUPR were much stronger than the event-only run. This supports the LMAT motivation that timing changes can expose abnormal behavior even when event identities remain similar.

The multi-task model gave the best overall balanced results. It achieved the strongest **average AUROC** and **average AUPR**, and it was also the best run on the **network** anomaly by a clear margin in F1. On memory stress it slightly outperformed the duration-only model in AUROC and AUPR, while still preserving high event-prediction accuracy.

## Main Takeaways

- The **multi-task LSTM** is the strongest overall category-5 LSTM run so far.
- The **duration signal** is important on this dataset; it clearly helps for memory-related anomalies.
- **Disk stress** appears to be the hardest anomaly type across all three LSTM setups.
- All three runs appear to plateau well before epoch 100, so later experiments could consider early stopping or shorter schedules to reduce compute cost.

## Recommended Table Direction For Later Extension

This file is structured so it can be extended later with:

- Transformer results
- additional duration-category settings such as 3, 7, and 9
- a final paper-style comparison table using only average AUROC/AUPR/F1 per model/category
