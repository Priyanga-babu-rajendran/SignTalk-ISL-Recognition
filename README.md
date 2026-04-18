# SignTalk – Indian Sign Language Recognition System
### Real-time ISL recognition using YOLOv5, SVM, and NLP

---

## 📌 Problem Statement
Sign language is the primary mode of communication for millions of hearing-impaired individuals. However, real-time translation systems are limited, often requiring expensive hardware or human interpreters.

This project aims to build an **accessible, real-time Indian Sign Language (ISL) recognition system** that can translate gestures into meaningful text and sentences.

---

## ⚙️ Methodology
The system is built as a **hybrid expert system** combining classical machine learning and deep learning:

### 🔹 YOLOv5 (Deep Learning)
- Real-time hand gesture detection and classification  
- Trained on a custom dataset of ISL gestures  
- Handles spatial and contextual understanding  

### 🔹 SVM + MediaPipe (Machine Learning)
- Extracts **21 hand landmarks (63 features)** using MediaPipe  
- Uses SVM with RBF kernel for classification  
- Efficient and fast for real-time prediction  

### 🔹 Expert System
- Combines predictions from YOLOv5 and SVM  
- Selects output based on confidence scores  
- Improves overall robustness and accuracy  

### 🔹 NLP Integration
- Uses **Gramformer** to convert predicted word sequences  
- Generates grammatically correct sentences  

---

## 📊 Dataset
- 82 ISL word classes  
- 809 original images  
- Augmented to 3× dataset size  
- Annotated using Roboflow  

---

## 📈 Results
- YOLOv5 Accuracy: ~97%  
- SVM Accuracy: ~96%  
- Expert System: Improved overall reliability  
- Real-time gesture recognition achieved  

---

## 🧠 System Architecture
The system integrates:
- YOLOv5 → Detection  
- MediaPipe → Landmark extraction  
- SVM → Classification  
- Expert System → Decision fusion  
- Gramformer → Sentence generation  

---

## 🚀 Key Contributions
- Hybrid AI system combining ML + DL  
- Real-time continuous word recognition  
- Sentence-level translation using NLP  
- Custom ISL dataset creation  

---

## 🛠️ Tech Stack
- Python  
- PyTorch (YOLOv5)  
- OpenCV  
- MediaPipe  
- Scikit-learn (SVM)  
- Gramformer (NLP)  
- Streamlit (UI)  

---

## 🔮 Future Work
- Expand dataset with more ISL vocabulary  
- Improve real-time performance  
- Deploy as web/mobile application
