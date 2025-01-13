# Multi-Label Emotion Detection Using Hugging Face Transformers

### Class  
**BUAN 6342: Natural Language Processing**

### Date  
**Spring 2024**

---

## Description
This project implements a multi-label emotion detection model using the Hugging Face Transformers library. The goal is to classify text data into multiple emotion categories simultaneously, such as anger, joy, fear, and trust. The system leverages parameter-efficient fine-tuning (IA3), custom loss functions, and advanced evaluation metrics to ensure accuracy and scalability.

The dataset is tokenized, preprocessed, and passed through a sequence classification model fine-tuned with IA3 configuration for efficient multi-label classification. Metrics like F1 score (micro and macro) and accuracy are used to evaluate performance. The training and evaluation process is logged with Weights & Biases (W&B) for better monitoring and reproducibility.

Key features include:
- Efficient multi-label classification using Hugging Face Transformers.
- Custom loss function (`BCEWithLogitsLoss`) with class weighting for imbalanced datasets.
- Advanced model quantization with 4-bit precision to optimize computational resources.
- Tokenization and preprocessing for raw text data using `AutoTokenizer`.
- Deployment-ready code for fine-tuning, evaluation, and predictions on custom datasets.

---

## Technologies Used
- **Hugging Face Transformers**
- **PyTorch**  
- **Weights & Biases (W&B)**  
- **Datasets**  
- **IA3 Parameter-Efficient Fine-Tuning**  
- **Google Colab**  

---

## Key Features
- **Multi-Label Emotion Detection**: Simultaneously classifies text into multiple emotion categories.  
- **Parameter-Efficient Fine-Tuning**: Uses IA3 to reduce memory and computational requirements while maintaining performance.  
- **Quantization**: Implements 4-bit quantization for faster training and inference.  
- **Custom Training Loop**: Includes a custom trainer with a weighted loss function to handle class imbalance.  
- **Metrics and Logging**: Tracks F1 scores, accuracy, and loss metrics using W&B.  
- **Tokenization and Preprocessing**: Handles data transformation for seamless input into the model.  
- **Dataset Integration**: Easily adapts to different datasets using Hugging Face `datasets` library.  

---

## Methods Used
1. **Model Fine-Tuning**:
   - Fine-tuned `google/gemma-1.1-2b-it` with IA3 for efficient multi-label classification.
   - Configured model quantization with BitsAndBytes for computational optimization.
2. **Custom Training Loop**:
   - Implemented a `CustomTrainer` class to calculate loss using `BCEWithLogitsLoss` with positive class weighting.
   - Evaluated on validation data for macro and micro F1 scores.
3. **Data Preprocessing**:
   - Tokenized text data using `AutoTokenizer` with truncation for compatibility with the model.
   - Handled label encoding and data formatting for Hugging Face `Trainer`.
4. **Evaluation**:
   - Used metrics like F1 (micro and macro) and accuracy for robust performance assessment.
5. **Logging**:
   - Integrated W&B for real-time tracking of training and evaluation metrics.
6. **Prediction**:
   - Processed a test dataset and exported predictions to CSV for further analysis.

---

## Key Results
- Achieved high F1 scores and accuracy for multi-label emotion detection.
- Optimized training with efficient IA3-based fine-tuning and 4-bit quantization.
- Successfully logged and monitored training progress using W&B.

---
