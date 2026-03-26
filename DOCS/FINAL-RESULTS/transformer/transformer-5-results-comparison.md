# Transformer Results Comparison: Paper 5 Duration Categories

This note compares the three Transformer runs trained on the LMAT microservice dataset with the paper-style **5 duration categories** setting. In the code and result folders this appears as `cats6`, because class `0` is reserved for padding/non-exit positions and the actual duration bins are `1..5`.

The comparison below is based on:

- `train_rank0.log` from each run for convergence and best validation behavior
- `ood_results.json` from each run for final OOD detection metrics

Runs compared:

- Event-only: [train_rank0.log](C:\workplace\adaptive_tracer\logs\transformer\transformer-5-event\train_rank0.log)
- Duration-only: [train_rank0.log](C:\workplace\adaptive_tracer\logs\transformer\transformer-5-duration\train_rank0.log)
- Multi-task: [train_rank0.log](C:\workplace\adaptive_tracer\logs\transformer\transformer-5-multitask\train_rank0.log)

## OOD Detection Summary

| Test set | Event AUROC | Event AUPR | Event F1 | Duration AUROC | Duration AUPR | Duration F1 | Multi-task AUROC | Multi-task AUPR | Multi-task F1 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| CPU | 0.6162 | 0.5990 | 0.6733 | 0.5846 | 0.5745 | 0.6667 | 0.6332 | 0.6127 | 0.6687 |
| Disk | 0.5361 | 0.5276 | 0.6766 | 0.5551 | 0.5563 | 0.6674 | 0.5675 | 0.5603 | 0.6670 |
| Memory | 0.6039 | 0.5981 | 0.6763 | 0.7536 | 0.7600 | 0.6670 | 0.7565 | 0.7471 | 0.6666 |
| Network | 0.7195 | 0.7533 | 0.6806 | 0.6488 | 0.6246 | 0.6667 | 0.7197 | 0.7146 | 0.7096 |
| **Average** | **0.6189** | **0.6195** | **0.6767** | **0.6355** | **0.6289** | **0.6669** | **0.6692** | **0.6587** | **0.6780** |

## Training Summary

| Run | Best val loss | Final train accuracy | Final epoch summary | Runtime |
| --- | ---:| --- | --- | --- |
| Event-only | 0.9633 | Event acc `94.45%` | `loss=0.9668`, `acc_e=94.45%` | about 7h 06m |
| Duration-only | 0.7452 | Duration acc `67.85%` | `loss=0.7562`, `acc_l=67.85%` | about 7h 05m |
| Multi-task | 0.8607 | Event acc `94.14%`, Duration acc `67.49%` | `loss=0.8717`, `acc_e=94.14%`, `acc_l=67.49%` | about 7h 15m |

## Short Discussion

All three Transformer runs trained stably and converged without obvious failures. The event-only Transformer achieved strong next-event accuracy, but its OOD ranking quality was mixed: network detection was clearly the strongest case, while disk and memory remained weaker.

The duration-only Transformer improved ranking substantially on **memory stress**, where it clearly outperformed the event-only run in both AUROC and AUPR. However, its F1 values were tightly clustered around threshold-driven operating points, which suggests the ranking signal improved more than the final tuned decision boundary.

The multi-task Transformer is the strongest overall run in this category-5 Transformer set. It achieved the best **average AUROC**, **average AUPR**, and **average F1**, and it was also the best run on **CPU**, **memory**, and **network** AUROC. The network anomaly is where the multi-task setup showed the clearest practical advantage, with the highest F1 among the three Transformer runs.

## Main Takeaways

- The **multi-task Transformer** is the strongest overall Transformer result for paper-style 5 duration bins.
- The **duration signal** again matters most for **memory anomalies**.
- **Disk stress** is the hardest anomaly type for all three Transformer setups.
- The event-only Transformer reached strong token accuracy, but that did not automatically translate into the best anomaly-ranking metrics.
- The Transformer runs improved slowly and continued finding slightly better checkpoints late in training, especially for duration and multi-task.

## Recommended Table Direction For Later Extension

This file is structured so it can be extended later with:

- LSTM versus Transformer comparison
- additional duration-category settings such as 3, 7, and 9
- a final paper-style summary table using average AUROC, AUPR, and F1 per model/category
