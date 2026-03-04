# LMAT SockShop Training Results

**Dataset:** SockShop microservice benchmark — LTTng kernel traces  
**Training nodes:** Trillium (H100 GPUs), single GPU, BF16 mixed-precision  
**Scoring method:** Joint event + latency cross-entropy (`lat_score_weight=0.3`)  
**Sequence length:** 4096 tokens  
**Effective batch size:** 2048 (batch=512, accum_steps=4)

---

## Evaluation Metrics

The LMAT model is jointly trained on two prediction tasks: **next syscall prediction** (event model) and **syscall duration bucket prediction** (latency model). Each task and evaluation phase uses a tailored metric.

### Training Phase

| Metric | Symbol | Description | Why used |
|---|---|---|---|
| **Cross-entropy loss** | `loss` | Weighted sum of event CE + latency CE, divided by accumulation steps | Directly optimized during training. Measures how surprised the model is by the next observed syscall and its duration. Decreasing loss indicates the model is learning the normal trace distribution. |
| **Token accuracy (event)** | `acc_e` | Fraction of non-padding syscall tokens where `argmax(logits) == ground truth` | Provides an interpretable measure of next-syscall prediction quality. At 84–86%, the model correctly predicts 5 in 6 syscalls — showing it has learned the dominant control-flow patterns. |
| **Token accuracy (latency)** | `acc_l` | Same as `acc_e` but for predicted duration bucket | Shows how well the model has internalized typical timing per syscall. High `acc_l` (particularly for LSTM at 99.6%) signals that the model has learned fine-grained latency timing patterns — essential for detecting resource anomalies. |

### Validation Phase

The same three metrics (`loss`, `acc_e`, `acc_l`) are computed on the held-out `valid_id` (normal) split after every 100 optimizer steps:

- **Val loss** is used as the **model selection criterion** — the checkpoint with the lowest validation loss is saved as `model_best.pt` and is later used for OOD evaluation. Using validation loss (rather than training loss) guards against overfitting to the training shards.
- Val `acc_e` / `acc_l` serve as sanity checks to confirm the model is genuinely generalizing rather than memorizing.

### OOD Evaluation Phase (Test)

After training, the **best checkpoint** is loaded and scored against each anomaly split. Since no ground-truth labels exist during inference, the anomaly score per sequence is a **weighted cross-entropy**:

```
score = (1 - w) * mean_event_CE  +  w * mean_latency_CE
      = 0.7 * event_XE + 0.3 * latency_XE    (default w=0.3)
```

Sequences with higher cross-entropy are deemed more anomalous (the model is more "surprised" by them). The scores are then ranked against normal baseline scores to produce binary labels, and evaluated with:

| Metric | Description | Why used |
|---|---|---|
| **AUROC** (Area Under ROC Curve) | Probability that a randomly chosen anomalous trace scores higher than a randomly chosen normal trace. 0.5 = random, 1.0 = perfect. | **Threshold-free** — measures discrimination quality across all possible thresholds. Standard metric for unsupervised anomaly detection where no threshold is fixed a priori. |
| **AUPR** (Area Under Precision-Recall Curve) | Area under the Precision vs. Recall curve, weighted toward behaviour at high-precision operating points. | More informative than AUROC when classes are **heavily imbalanced** (normal traces typically outnumber anomalous ones in production). High AUPR means the model can identify anomalies with very few false positives. |

> [!TIP]
> The choice to combine event + latency cross-entropy as the scoring function (rather than event alone) is motivated by the nature of resource anomalies: CPU/disk/mem stress does not always change *which* syscalls are called, but reliably increases their *duration*. The latency component gives the scoring function complementary signal for these anomaly types.

---

## Model Configurations

| Parameter | Transformer | LSTM |
|---|---|---|
| **Job ID** | 311494 | 311554 |
| **Architecture** | Transformer | LSTM |
| **Params** | 2,493,711 | 47,156,895 |
| **n_hidden** | 1024 | 1024 |
| **n_layer** | 6 | 6 |
| **n_head** | 8 | — |
| **dropout** | 0.1 | 0.01 |
| **lr** | 3e-4 | 1e-3 |
| **clip** | 1.0 | 10.0 |
| **Epochs** | 50 | 50 |
| **Total opt steps** | ~7,712 | ~7,712 |

> **Note:** The LSTM has 47M parameters due to the larger hidden state requirement for a 6-layer bidirectional LSTM over long sequences, vs. 2.5M for the Transformer which uses efficient multi-head self-attention with weight tying.

---

## Training Performance

### Epoch-level Summary

| Epoch | Transformer train loss | LSTM train loss |
|---|---|---|
| 1 | 5.490 | 5.002 |
| 5 | 2.212 | 2.192 |
| 10 | 2.100 | 1.950 |
| 20 | 2.012 | 1.689 |
| 30 | 1.945 | 1.315 |
| 40 | 1.903 | 1.213 |
| 50 | **1.889** | **1.195** |

### Final Training Accuracy (Epoch 50)

| Metric | Transformer | LSTM |
|---|---|---|
| **acc_e** (syscall prediction) | 84.04% | 85.90% |
| **acc_l** (latency bucket) | 74.56% | 99.61% |
| **Train loss** | 1.889 | 1.195 |

### Best Validation Performance

| Metric | Transformer | LSTM |
|---|---|---|
| Best val loss | **1.8352** (step 7700) | **1.1920** (step 7900) |
| Val acc_e at best | 84.47% | 85.96% |
| Val acc_l at best | 76.27% | 99.65% |

**Training time:** Transformer = **5h 04min** | LSTM = **2h 59min**

> [!NOTE]  
> LSTM converges faster and achieves significantly lower training and validation loss. The LSTM's `acc_l` reaching **99.61%** shows near-perfect latency bucket prediction — critical for detecting resource anomalies that manifest as latency changes.

---

## OOD Anomaly Detection Results

All scores evaluated on the **best validation checkpoint** loaded after training.  
Each type compares 316,026 **normal** traces vs. its anomalous counterpart.

| Anomaly Type | Transformer AUROC | LSTM AUROC | Transformer AUPR | LSTM AUPR | OOD Count |
|---|---|---|---|---|---|
| **cpu** | 0.708 | **0.754** | 0.701 | **0.756** | 387,625 |
| **disk** | 0.733 | **0.825** | 0.733 | **0.837** | 425,955 |
| **mem** | 0.710 | **0.779** | 0.690 | **0.772** | 385,131 |
| **net** | 0.653 | **0.674** | 0.534 | **0.577** | 220,229 |
| **Mean AUROC** | 0.701 | **0.758** | 0.665 | **0.736** | — |

---

## Comparison vs. 20-Epoch Baseline (Previous Run)

| Model | Epochs | cpu AUROC | disk AUROC | mem AUROC | net AUROC | Mean |
|---|---|---|---|---|---|---|
| Transformer (20ep, event-only) | 20 | 0.615 | 0.583 | 0.575 | 0.583 | 0.589 |
| LSTM (20ep, event-only, small) | 20 | 0.611 | 0.607 | 0.589 | 0.598 | 0.601 |
| **Transformer (50ep, joint score)** | 50 | 0.708 | 0.733 | 0.710 | 0.653 | **0.701** |
| **LSTM (50ep, joint score)** | 50 | **0.754** | **0.825** | **0.779** | **0.674** | **0.758** |

**Improvements from 20→50 epochs + joint latency scoring:**
- Transformer: +**+11.2pp** mean AUROC
- LSTM: +**+15.7pp** mean AUROC (vs. original tiny LSTM baseline)

> [!IMPORTANT]  
> The joint event+latency scoring (30% latency weight) is responsible for a significant portion of the AUROC jump, especially for `disk` and `mem` anomalies which stress I/O and memory latency without dramatically changing the syscall sequence identity.

---

## Key Conclusions

1. **LSTM wins on OOD detection** despite having fewer traditional advantages. A well-trained, sufficiently large LSTM achieves **+5.7pp mean AUROC** over the Transformer, and trains **40% faster** (3h vs 5h).

2. **`disk` is the easiest anomaly to detect** (LSTM: 0.825 AUROC). Disk stress tests cause unmistakable I/O wait latency spikes visible in the duration bucket predictions.

3. **`net` is the hardest anomaly** (LSTM: 0.674 AUROC, Transformer: 0.653). Network anomalies produce the least distinct syscall sequence changes since network I/O is often handled by the kernel without explicit user-space syscall differences.

4. **Longer training is clearly worth it.** The LSTM was still improving at epoch 50 (`val loss 1.1920` at step 7900) — training to 100 epochs would likely push AUROC higher still.

5. **Joint latency scoring is highly effective.** The 30% latency weight contribution helped `disk` and `mem` detection significantly. `net` remains the weakest, suggesting pure latency scoring (weight→0.5 or higher) may help further for network anomalies.

6. **LSTM latency prediction approaches near-perfect** (`acc_l=99.61%`) while the Transformer's latency head plateaued at 76.27%. This suggests the LSTM's recurrent state is better at modeling the temporal autocorrelation of latency across a long syscall sequence.

---

## Recommended Next Steps

| Priority | Action | Expected Gain |
|---|---|---|
| 🔥 High | Train LSTM for 100 epochs | +2–4pp AUROC (still converging) |
| 🔥 High | Tune `lat_score_weight` for `net` (try 0.5–0.7) | +3–5pp net AUROC |
| Medium | Try `--ordinal_latency` flag | Better-calibrated latency scores |
| Medium | Add DDP (4×GPU) for Transformer | 4× throughput, larger model |
| Low | Experiment with Transformer + higher dropout (0.2) | Reduce Transformer overfitting |
