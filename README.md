# Comparing RGB and Wavelet Representations for CNN-Based Deepfake Face Detection

MSc thesis project. GAN-generated faces can look real to the eye but leave traces in the high
frequencies of an image. This project tests whether a CNN detects them better from the RGB image,
from wavelet features, or from both.

## Method

- **Data:** [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) on Kaggle.
  Real faces come from FFHQ, fake faces from StyleGAN. 256×256 images, split 100,000 train / 20,000 validation / 20,000 test.
- **Inputs:**
  - RGB, 3 channels
  - Haar wavelet: the three high-frequency sub-bands (LH, HL, HH) of each colour channel, 9 channels, standardised
  - Combined: both stacked, 12 channels
- **Model:** three DenseNet-121 networks trained from scratch with the same head and the same training setup.
  Adam with weight decay, ReduceLROnPlateau, dropout and early stopping. Horizontal flips and Gaussian noise
  on the training data only.
- **Evaluation:** accuracy, precision, recall, F1 and AUC on the test set; mean ± std over 10 stratified test
  subsets; bootstrap confidence intervals; McNemar's test between the RGB and the combined model; Grad-CAM
  to see where each model looks.

## Results

Test set, mean ± std over 10 stratified subsets of 2,000 images:

| Input | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| RGB | 97.69 ± 0.33% | 97.87 ± 0.50% | 97.50 ± 0.39% | 97.69 ± 0.33% | 0.997 |
| Wavelet | 97.07 ± 0.44% | 98.53 ± 0.44% | 95.56 ± 0.52% | 97.02 ± 0.45% | 0.997 |
| Combined | **98.12 ± 0.22%** | 98.46 ± 0.37% | 97.76 ± 0.40% | **98.11 ± 0.22%** | **0.999** |

- The combined input is significantly better than RGB alone: McNemar's χ² = 9.71, p = 0.0018 on the 20,000 test images.
- The wavelet model has the highest precision for "real": it is the most conservative about calling a fake image real.
- Grad-CAM: the RGB model focuses on facial features, the wavelet model on edges and boundaries,
  and the combined model on several regions at once.

## Files

- `deepfake_detection.ipynb`: the full pipeline, with the outputs of the original run on Google Colab (NVIDIA L4 GPU).

The trained weights (about 90 MB per model) are not included. With `TRAIN = True` the notebook trains
the models again; otherwise it loads the weights from `weights/`.

## Run

```bash
pip install -r requirements.txt
```

Then open the notebook. The dataset is downloaded with `kagglehub`, which needs a Kaggle account.

## More

Course projects from the same MSc: [data-science-course-projects](https://github.com/HJLin057/data-science-course-projects)
