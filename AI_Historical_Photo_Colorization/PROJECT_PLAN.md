# Project Development Plan

## Week 1 — Data Preparation

1. Download the Kaggle dataset.
2. Extract images into `dataset/`.
3. Run preprocessing through `train.py`.
4. Verify image dimensions.
5. Verify grayscale input and RGB target.

## Week 2 — Deep Learning Model

1. Build CNN Autoencoder.
2. Train for 15–30 epochs.
3. Monitor training and validation loss.
4. Save the best model.
5. Capture model summary for the weekly submission.

## Week 3 — Testing and Evaluation

1. Prepare several black-and-white test images.
2. Run `predict.py`.
3. Compare grayscale and predicted color.
4. Record test loss and MAE.
5. Add PSNR/SSIM later if required.
6. Save screenshots for the report.

## Week 4 — Final Demo

1. Run `streamlit run app.py`.
2. Upload a historical photograph.
3. Generate colorized result.
4. Add before/after screenshots to PPT.
5. Prepare architecture and workflow diagrams.
6. Prepare final report and demonstration.

## Presentation Points

### Problem
Historical photographs are often available only in grayscale, making visual interpretation less vivid.

### Proposed Solution
A CNN Autoencoder learns color patterns from RGB images and predicts plausible RGB colors from grayscale input.

### Input
Black-and-white photograph.

### Output
AI-generated color photograph.

### Limitation
The predicted colors are plausible rather than guaranteed to be the exact historical colors.
