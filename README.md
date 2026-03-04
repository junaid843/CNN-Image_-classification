# 🧠 NeuralForge — Custom CNN Image Classifier

A full end-to-end machine learning web application built with **TensorFlow/Keras** and **Streamlit**.

---

## 📁 Project Structure

```
ml_webapp/
├── app.py              ← Streamlit UI (main entry point)
├── train_model.py      ← CNN model, training pipeline, prediction
├── requirements.txt    ← Python dependencies
├── dataset/            ← Auto-created; holds class subfolders
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── models/             ← Auto-created; stores trained model & logs
    ├── best_model.keras
    ├── class_names.json
    ├── training_log.json
    ├── eval_results.json
    └── confusion_matrix.png
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```

### 3. (Optional) Train from command line
```bash
python train_model.py --dataset dataset --epochs 30 --val_split 0.2
```

---

## 🚀 Features

### 📂 Dataset Tab
- Create unlimited custom classes (Cat, Dog, Hand Gesture, etc.)
- Upload images in bulk via drag-and-drop
- Capture images using webcam
- Preview uploaded images in a grid
- Dataset health dashboard (minimum 30 images per class recommended)

### 🏋️ Train Tab
- Configure epochs and validation split
- Live training progress (accuracy + loss curves via Plotly)
- Early stopping & best-model checkpointing
- Background subprocess training — UI stays responsive

### 📊 Evaluate Tab
- Final accuracy, Macro F1 score
- Confusion matrix heatmap
- Per-class precision / recall / F1

### 🔍 Predict Tab
- Upload any image → instant prediction
- Confidence score + probability bar chart for all classes
- Optional webcam capture → predict

---

## 🏗 CNN Architecture

```
Input (128×128×3)
  ↓
Conv2D(32) → BatchNorm → ReLU
Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  ↓
Conv2D(64) → BatchNorm → ReLU
Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  ↓
Conv2D(128) → BatchNorm → ReLU
Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.40)
  ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.50)
  ↓
Softmax (N classes)
```

- **Optimizer**: Adam (lr=1e-3, with ReduceLROnPlateau)
- **Augmentation**: rotation, shift, flip, zoom, shear
- **Regularization**: BatchNorm + Dropout at every block

---

## 📋 Tips

| Tip | Details |
|-----|---------|
| Minimum images | 30–50 per class for decent accuracy |
| More epochs | Try 30–50 for better convergence |
| Overfitting | Watch train vs val accuracy gap |
| Image quality | Consistent backgrounds help |
