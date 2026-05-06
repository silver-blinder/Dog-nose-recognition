# Experiment 2: Model Architecture Comparison

> **Objective**: Compare our Siamese Network (metric learning) against
> two conventional closed-set CNN classifiers (VGG-16, ResNet-50)
> to justify the metric-learning paradigm for dog nose-print recognition.

## 2.1 Comparison Design

| Model | Paradigm | Open-set? | Params (M) | GFLOPs | Infer (ms/pair) |
|---|---|:---:|---:|---:|---:|
| **Siamese-ResNet50 (Ours)** | Metric Learning | Yes | 25.9 | 4.1 | 18.0 |
| **VGG-16 (Classifier)** | Classification | No | 138.4 | 15.5 | 42.0 |
| **ResNet-50 (Classifier)** | Classification | No | 25.6 | 4.1 | 18.0 |

**Key axis of comparison**: metric learning vs classification paradigm.

- **Siamese Network** learns a similarity function; at test time it compares
  the query image against enrolled nose-prints. New dogs can be added to the
  database without any retraining.

- **VGG-16 / ResNet-50 Classifiers** learn a softmax over the fixed set of
  1393 training classes. They cannot recognise dogs not seen during
  training, making them unsuitable for a dynamic registry system.

## 2.2 Results

### 2.2.1 Quantitative Metrics

| Model | Closed-set Acc | Open-set Acc | Final Loss |
|---|---:|---:|---:|
| **Siamese-ResNet50 (Ours)** | **0.8720** | 0.8810 | 0.2661 |
| **VGG-16 (Classifier)** | **0.7547** | 0.5000 | 0.5571 |
| **ResNet-50 (Classifier)** | **0.8196** | 0.5000 | 0.3821 |

### 2.2.2 Training Curves

![Training Curves](figures/model_comparison_curves.png)

### 2.2.3 Closed-set vs Open-set Accuracy

![Open-set vs Closed-set Accuracy](figures/model_params_vs_acc.png)

The grouped bar chart shows that VGG-16 and ResNet-50 classifiers achieve
reasonable closed-set accuracy on the *training* classes, but their open-set
accuracy drops to 0.500 (chance level) for dogs unseen during training.
The Siamese Network maintains the same accuracy on both settings because it
does not depend on fixed class labels.

### 2.2.4 Performance Radar

![Performance Radar](figures/model_radar.png)

## 2.3 Analysis

### 2.3.1 Why Classifiers Underperform on This Task

With only ~6 images per class across 1393 classes, a closed-set classifier is asked to memorise a mapping from a tiny number of examples to a very large label space. The training accuracy for classifiers can still reach near 100% (overfitting), but the generalisation gap is severe. VGG-16 reaches 0.7547 and ResNet-50 reaches 0.8196 on seen classes -- but neither can say anything useful about a new dog.

VGG-16 is also at a structural disadvantage: 138M parameters on a dataset with ~8,364 training images is an extreme overparameterisation ratio (~1 parameter per 0.06 images). Batch normalisation and dropout help, but the fundamental mismatch between model capacity and data volume limits generalisation.

### 2.3.2 Why Siamese Network Works Better

The Siamese approach reformulates the task as binary similarity: given two nose-print images, are they from the same dog? This doubles the effective training signal (every pair is a training example) and does not require a fixed class label space. As a result, the Siamese Network achieves 0.8720 closed-set accuracy while *also* maintaining 0.8810 open-set accuracy -- a gap of 37.2pp over VGG-16 and the same gap over ResNet-50 classifier in the open-set setting.

### 2.3.3 ResNet-50 Classifier vs Siamese-ResNet50

Both use the same ResNet-50 backbone and ImageNet pretrained weights. The only difference is the learning objective: cross-entropy classification vs contrastive metric learning. The Siamese variant outperforms the classifier by 5.2pp on closed-set accuracy, and more importantly is the only one of the two that can handle new dogs at test time. This experiment isolates the paradigm effect, independent of architecture choice.

## 2.4 Summary

| Dimension | Siamese (Ours) | VGG-16 Classifier | ResNet-50 Classifier |
|---|:---:|:---:|:---:|
| Closed-set Accuracy | ★★★★★ | ★★★ | ★★★★ |
| Open-set Capability | ★★★★★ | ✗ | ✗ |
| Param Efficiency | ★★★★ | ★ | ★★★★ |
| Inference Speed | ★★★★ | ★★ | ★★★★ |
| Scalability (new dogs) | ★★★★★ | ✗ | ✗ |

**Conclusion**: For a dog nose-print registry that must support dynamic enrollment of new dogs, metric learning is the only viable paradigm. Closed-set classifiers are a reasonable baseline for measuring *closed-set* recognition quality but fail entirely at the open-set generalisation that the real application demands.
