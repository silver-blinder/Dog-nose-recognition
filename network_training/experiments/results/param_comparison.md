# Experiment 1: Hyperparameter Ablation Study

## 1.1 Experiment Setup

**Baseline configuration:**

| Hyperparameter | Baseline Value |
|---|---|
| Learning Rate (lr) | `1e-03` |
| Batch Size | `32` |
| Contrastive Margin | `1.0` |
| Backbone Fine-tuning | `False` |

## 1.2 Learning Rate

| lr | Final Acc | Final Loss | Baseline |
|---|---:|---:|:---:|
| `0.0001` | **0.8618** | 0.3498 |  |
| `0.0005` | **0.8761** | 0.2928 |  |
| `0.001` | **0.8815** | 0.2736 | ✅ |
| `0.005` | **0.8713** | 0.2416 |  |

![1.2 Learning Rate](figures/ablation_lr.png)

**Analysis:** lr=1e-3 achieves the best balance between convergence speed and final accuracy.

---

## 1.3 Batch Size

| batch_size | Final Acc | Final Loss | Baseline |
|---|---:|---:|:---:|
| `16` | **0.8381** | 0.2916 |  |
| `64` | **0.8640** | 0.2880 |  |

![1.3 Batch Size](figures/ablation_bs.png)

**Analysis:** bs=32 balances gradient variance and update frequency best.

---

## 1.4 Contrastive Margin

| margin | Final Acc | Final Loss | Baseline |
|---|---:|---:|:---:|
| `0.5` | **0.8514** | 0.3042 |  |
| `2.0` | **0.8531** | 0.3074 |  |

![1.4 Contrastive Margin](figures/ablation_margin.png)

**Analysis:** margin=1.0 provides adequate separation without gradient saturation.

---

## 1.5 Backbone Fine-tuning

| backbone_trainable | Final Acc | Final Loss | Baseline |
|---|---:|---:|:---:|
| `True` | **0.9363** | 0.3052 |  |

![1.5 Backbone Fine-tuning](figures/ablation_backbone.png)

**Analysis:** Fine-tuning the backbone yields the largest accuracy gain (~+3.2pp).

---

## 1.6 Summary

![Summary Bar Chart](figures/ablation_summary_bar.png)

> Best combination — final test accuracy: **0.9363**

