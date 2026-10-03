# 🕰️ HistoricColor AI

## AI-Based Historical Photo Colorization Using Autoencoder

This project uses a CNN-based Autoencoder to convert black-and-white photographs into plausible color images.

### Workflow

Grayscale Image → CNN Encoder → Latent Features → CNN Decoder → RGB Color Image

## Folder Structure

```text
AI_Historical_Photo_Colorization/
├── dataset/
├── models/
├── outputs/
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install packages

Open CMD/PowerShell inside this folder:

```bash
pip install -r requirements.txt
```

## 2. Add dataset

Download a color image dataset and put JPG/PNG images directly inside:

```text
dataset/
```

Recommended Kaggle datasets:

- Landscape Image Colorization:
  https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization

- Image Colorization:
  https://www.kaggle.com/datasets/balraj98/image-colorization

Do not put the ZIP file itself in `dataset/`; extract the images first.

## 3. Train

```bash
python train.py
```

The best trained model will be saved as:

```text
models/colorization_autoencoder.keras
```

For a first test, you can change `EPOCHS = 30` in `train.py` to `5`.

## 4. Test one image

Put a black-and-white image in the project root and rename it:

```text
test.jpg
```

Then run:

```bash
python predict.py
```

The result is saved in:

```text
outputs/colorized_result.png
```

## 5. Run the web application

```bash
streamlit run app.py
```

The browser interface will allow you to upload an image and generate a colorized result.

## Important

The model predicts plausible colors. It does not recover the exact original historical colors because grayscale images do not contain that information.

## Suggested 4-week plan

Week 1:
- Dataset
- Preprocessing
- RGB → grayscale pairs

Week 2:
- Autoencoder architecture
- Training
- Loss graphs

Week 3:
- Testing
- Before/after results
- MAE/MSE/PSNR evaluation

Week 4:
- Streamlit application
- Final screenshots
- PPT and report
- Demo

## Project title

AI-Based Historical Photo Colorization Using Autoencoder

## Suggested application name

HistoricColor AI
