# Handwritten Character Recognition using CNN

This project is aimed at recognizing handwritten English characters (A–Z, a–z, 0–9) using a deep learning model. It is trained on the EMNIST ByClass dataset and achieves an accuracy of around **85–87%** on the validation set.

The model is built using PyTorch and leverages convolutional layers to automatically extract spatial features from grayscale 28x28 pixel images.

---

## Features

- Recognizes 62 character classes: uppercase letters, lowercase letters, and digits  
- Trained using Convolutional Neural Network (CNN)  
- Achieves 85–87% accuracy on validation data  
- Generates confusion matrix and accuracy/loss plots  
- Saves the trained model for future use

---

## Dataset Used

- **EMNIST ByClass** from `torchvision.datasets`  
- Contains 814,255 grayscale 28x28 handwritten character images  
- Covers 62 classes (0–9, A–Z, a–z)  
- Dataset is automatically downloaded when running the code

---

## Model Architecture

- 2× Conv2D layers with ReLU and MaxPooling  
- Fully Connected Layer with ReLU  
- Final Dense output layer with 62 nodes (softmax)  
- Optimizer: Adam  
- Loss Function: CrossEntropyLoss  
- Framework: PyTorch

---

## Future Enhancements

- Add dropout and batch normalization layers  
- Extend to real-time handwriting recognition using webcam  
- Build a simple Streamlit web interface  
- Train on additional datasets for multilingual support  
- Implement character sequence prediction (e.g., full words)

---

## IMPORTANT
Extendable to full word or sentence recognition using sequence models like CRNN (Convolutional Recurrent Neural Network)

---

## Tech Stack

- Python  
- PyTorch  
- NumPy  
- Matplotlib  
- scikit-learn  
- Torchvision

---



## How to Run

1. Clone the repository  
2. Make sure Python and pip are installed  
3. Install required packages: 
   ```bash
   pip install -r requirements.txt
4. Run the training script:
   ```bash
   python handwritten_character_project.py
5. Wait for training to complete:
   Outputs:
- Confusion Matrix
- Accuracy & Loss plots


## License
This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
See the [LICENSE](LICENSE) file for details.
