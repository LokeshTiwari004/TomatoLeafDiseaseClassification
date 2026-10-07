# Tomato Leaf Disease Classification

A PyTorch Lightning + ResNet-50 transfer learning pipeline for multi-class classification of tomato leaf diseases using the PlantVillage dataset.

## Key Features

- **Leakage-safe cross-validation:** Images from the same leaf (`leaf_id`) are kept in the same split using `StratifiedGroupKFold`.
- **Staged fine-tuning:** Backbone starts frozen, then is unfrozen at 25% of training steps to stabilise optimisation.
- **Targeted data augmentation:** Colour jitter, horizontal/vertical flips, and ±15° rotations applied only during training.
- **Imbalance-aware evaluation:** Macro-averaged F1 drives checkpoint selection and is reported alongside per-class F1.
- **Reproducible setup:** Central `CFG`, fixed seed, persisted Parquet splits and label mappings.

## Fine-Tuning Approach

This work adopts transfer learning with a pretrained ResNet-50 (He et al., 2016) initialised on ImageNet. Building on evidence that shallow layers generalise across domains while deeper layers specialise (Yosinski et al., 2014), a staged adaptation strategy is used:

- **Head-only initialisation:** The backbone is frozen and the final fully connected layer is replaced with a task-specific linear head. This allows the classifier boundary to adapt without distorting generic pretrained features.
- **Staged unfreezing:** At 25% of total training steps, the backbone is unfrozen via the `UnfreezeBackbone` callback. This gradual release limits catastrophic forgetting while letting deeper layers specialise to tomato leaf disease features (Howard & Ruder, 2018; Yosinski et al., 2014).
- **Warm-up + cosine decay:** AdamW (Loshchilov & Hutter, 2017) is optimised with linear warm-up over 15% of training steps followed by cosine annealing (Loshchilov & Hutter, 2016) to improve early stability.
- **Domain-appropriate augmentation:** Training samples use colour jitter, horizontal/vertical flips, and random rotations (±15°) to promote pose and illumination invariance without obscuring disease cues (Shorten & Khoshgoftaar, 2019). Validation and test samples use only dtype scaling and ImageNet normalisation.
- **Leakage prevention:** `StratifiedGroupKFold` grouped by `leaf_id` preserves class stratification while ensuring all images from a given physical leaf remain in the same fold (Pedregosa et al., 2011).
- **Macro-F1–based selection:** Macro-averaged F1 (Sokolova & Lapalme, 2009) is used for checkpointing to weight all classes equally. Per-class F1 and out-of-fold (OOF) macro-F1 (mean ± SD) are reported across all folds.

### References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778. https://doi.org/10.1109/CVPR.2016.90
- Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. *ACL*, 328–339. https://doi.org/10.18653/v1/P18-1031
- Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health. *arXiv*:1511.08060. https://arxiv.org/abs/1511.08060
- Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic gradient descent with warm restarts. *ICLR*. https://arxiv.org/abs/1608.03983
- Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *ICLR*. https://arxiv.org/abs/1711.05101
- Pedregosa, F. et al. (2011). Scikit-learn: ML in Python. *JMLR*, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Shorten, C., & Khoshgoftaar, T. M. (2019). Image data augmentation for deep learning. *J. Big Data*, 6(1), 60. https://doi.org/10.1186/s40537-019-0197-0
- Sokolova, M., & Lapalme, G. (2009). Performance measures for classification tasks. *IP&M*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *NeurIPS*, 27. https://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks

## Dataset

- **Source:** [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) (color images)
- **Subset:** Tomato classes only

## Requirements

```text
torch
torchvision
pytorch-lightning
torchmetrics
datasets
scikit-learn
pyarrow
pandas
matplotlib
kaggle-secrets  # only if running on Kaggle
```

## Usage

1. Clone this repository.
2. Open `tomatoleafdisease (1).ipynb` in Kaggle (recommended, T4 GPU) or locally.
3. Set `HF_TOKEN` via Kaggle secrets or environment variable.
4. Run all cells to generate splits, train folds, and compute OOF metrics.

## Results

Final evaluation reports the test macro-F1 (mean ± SD) across folds and per-class OOF F1 at the end of the notebook.

## Citation

If you use this work, please cite the PlantVillage dataset: Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint* arXiv:1511.08060. https://arxiv.org/abs/1511.08060
