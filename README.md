# ✍️ Task 2 - Handwritten Character Recognition (A–Z)

This project was completed as part of the **CodeAlpha Internship Program (June 2025)**.  
It involves training a CNN model using the **EMNIST (ByClass)** dataset to recognize **handwritten English letters and digits**.

---

## 🎯 Objective

To classify handwritten characters from the EMNIST dataset into one of **62 classes**:
- Digits: `0–9`
- Uppercase: `A–Z`
- Lowercase: `a–z`

---

## 📊 Accuracy Achieved

✅ **Final Validation Accuracy:** 85–87%  
✅ Evaluated using **Confusion Matrix** and **Accuracy/Loss graphs**

---

## 🧠 Model Architecture (PyTorch)

- `Conv2D` → `ReLU` → `MaxPool`
- `Conv2D` → `ReLU` → `MaxPool`
- `Flatten` → `Linear` → `ReLU` → `Linear (output)`
- Output shape: `62` classes

---

## 🧪 Features

- Uses **EMNIST ByClass** dataset (62 classes)
- CNN-based character recognition model
- Proper image rotation + flipping to match EMNIST structure
- Visualizations: character samples, accuracy/loss over epochs, confusion matrix

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com//kanishka9r/codealpha_handwritten_character_recognition.git
cd codealpha_handwritten_character_recognition