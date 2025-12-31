# Explainable AI for Criminal Verdict Prediction

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

## Executive Summary
This app is an end-to-end Machine Learning pipeline designed to analyze unstructured Indonesian criminal court decisions (*Putusan Pidana*). By leveraging **Bi-Directional LSTMs** and **Feature Injection techniques**, the system predicts imprisonment duration based on case narratives.

Critically, this project integrates **Explainable AI (LIME)** to ensure transparency, allowing legal practitioners to visualize which specific words (e.g., *residivis*, *sopan*) influenced the model's prediction. This project serves as a foundational step towards building Trustworthy AI in the legal tech domain.

---

## Business Context & Problem Statement
The Supreme Court of Indonesia processes approximately **100,000 new documents monthly**. Legal practitioners and judicial researchers face a significant "Information Overload" bottleneck. Analyzing consistency in sentencing across thousands of PDF documents is computationally expensive and prone to human error.

**Objective:**
To build a **Data-Centric AI solution** that automates the extraction of legal reasoning and predicts sentencing outcomes, serving as a decision-support system for legal consistency.

---

## 🛠️ Technical Architecture

### 1. Data Pipeline & Robust Preprocessing
*Relevance: Data Validation & Quality Assurance*

The raw data consists of PDF documents containing OCR artifacts, watermarks, and inconsistent formatting. A rigorous cleaning pipeline was implemented to ensure data quality:
* **Noise Removal:** Regex-based removal of "phantom characters" (e.g., watermark *maa maa*), headers, and footers.
* **Entity Extraction:** Automated extraction of key legal entities:
    * *Articles Charged* (Pasal)
    * *Prosecutor's Demand* (Tuntutan)
    * *Mitigating & Aggravating Factors* (Hal Meringankan/Memberatkan)
* **Feature Injection:** These extracted entities are concatenated with the case narrative to strictly guide the model's attention mechanism.

### 2. Modeling Strategy (Bi-LSTM)
While Large Language Models (LLMs) are the current state-of-the-art, this project utilizes a **Bi-Directional LSTM** to establish a highly efficient, low-latency baseline.
* **Embedding Layer:** Learned dense vector representations of legal terminology.
* **Bi-Directional Wrapper:** Captures sequential context from both past and future tokens, essential for understanding legal negation (e.g., *"Terdakwa **tidak** terbukti..."*).
* **Regression Output:** Predicts a continuous value (months of imprisonment).

### 3. Explainability (XAI)
To address the "Black Box" nature of Neural Networks in high-stakes domains, **LIME (Local Interpretable Model-agnostic Explanations)** is implemented.
* **Green Highlights:** Words increasing the sentence duration.
* **Red Highlights:** Words decreasing the sentence duration.

---

## 📂 Repository Structure

```text
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── notebooks/
│   └── analysis.ipynb       # Model training & exploration (Jupyter)
├── models/
│   ├── model_sentence_prediction_lstm.h5  # Trained Model
│   └── tokenizer.pickle     # Saved Tokenizer for consistent inference
├── utils/
│   └── preprocessing.py     # Cleaning & extraction functions
├── data/
│   └── sample_cases/        # Sample PDF documents for testing
└── README.md                # Project documentation