# Product Review Sentiment Analysis Using LSTM

## Project Overview

This project focuses on building a sentiment classification model using Long Short-Term Memory (LSTM) networks. We classify user product reviews as either **Positive** or **Negative**, using deep learning techniques from the Keras/TensorFlow ecosystem.

## Objective

To develop a binary classifier that can automatically predict the sentiment of product reviews using sequence-based modeling (LSTM), allowing for accurate understanding of user opinions.

## Dataset

- Source: Real-world `Reviews.csv` dataset containing product review text and sentiment labels.
- Original Size: ~500,000 rows
- Preprocessing:
  - Only reviews with binary sentiments were retained.
  - Balanced dataset: Equal number of positive and negative reviews were sampled for training fairness.
  - Final shape after cleaning: ~65,000 rows (50/50 positive vs negative)

## Workflow Summary

### 1. Preprocessing
- Cleaned review text (lowercasing, punctuation removal).
- Converted sentiment labels to binary (1: Positive, 0: Negative).
- Balanced the dataset using random undersampling.
- Split into training and testing sets (80/20).

### 2. Text Vectorization
- Tokenized text using Keras `Tokenizer`.
- Converted sequences to padded arrays of equal length (200 words max).

### 3. Model Architecture (LSTM)
A simple sequential deep learning model:
- `Embedding` Layer – Converts word indices to vector representations.
- `LSTM` Layer – Captures sequential patterns and dependencies.
- `Dropout` – Regularization to avoid overfitting.
- `Dense` Layer – Fully connected layer for learning.
- `Output` – Sigmoid-activated neuron for binary classification.

### 4. Training
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- EarlyStopping was used to halt training when validation loss stopped improving.
- Trained for 5 epochs with batch size 128.

### 5. Evaluation

- **Test Accuracy:** ~94%
- **Precision/Recall/F1-Score:** Balanced across both classes.
- **Confusion Matrix:** High true positive and true negative counts, minimal misclassification.

## Key Findings

- LSTM networks perform well on text-based sentiment classification tasks.
- Even with a simple model architecture and limited preprocessing, strong results were achieved (~94% accuracy).
- GlobalAveragePooling can be swapped with LSTM to better capture word sequence patterns, improving performance for longer or more context-sensitive reviews.

## Requirements

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- pandas, numpy, matplotlib, seaborn

## Future Improvements

- Incorporate pre-trained embeddings (e.g., GloVe).
- Add BiLSTM for bidirectional context.
- Experiment with attention mechanisms or Transformer models (like BERT).
- Create a deployable web app or API for real-time review scoring.

## Repository Structure

- `data/` – Input CSV file.
- `notebooks/` – Jupyter notebooks used during experimentation.
- `model/` – Final model architecture and weights (optional).
- `README.md` – Project summary and documentation.

## License

This project is open-source and available under the MIT License. Tested and Approved
