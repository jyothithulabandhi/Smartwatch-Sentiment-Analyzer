# ⌚ Smartwatch Sentiment Analyzer

A Natural Language Processing (NLP) based web application that analyzes customer reviews of smartwatches and classifies them into **Positive**, **Neutral**, or **Negative** sentiments using both **Classical Machine Learning** and **Transformer-based Deep Learning models**.

---

##  Project Overview

With the rapid growth of e-commerce platforms, customers leave millions of product reviews daily. Manually analyzing these unstructured reviews is inefficient and error-prone.  
The **Smartwatch Sentiment Analyzer** automates this process by intelligently understanding customer opinions, including **context, sarcasm, negation, and mixed sentiments**.

This project combines:
- **Traditional ML models (Naive Bayes + TF-IDF)** for fast and interpretable results
- **Transformer models (BERT / RoBERTa)** for deep contextual understanding



## Objectives

- Automatically classify smartwatch reviews into sentiment categories
- Compare classical ML models with transformer-based models
- Perform **sentence-level sentiment analysis** for mixed reviews
- Deploy the system as a **Flask-based web application**



## Technologies Used

### Programming & Frameworks
- Python 3.8+
- Flask (Web Framework)

### NLP & Machine Learning
- NLTK
- scikit-learn
- TF-IDF Vectorizer
- Naive Bayes Classifier
- Logistic Regression
- Random Forest

### Deep Learning
- Hugging Face Transformers
- BERT / RoBERTa

### Data Handling
- Pandas
- NumPy


## Project Structure

│
├── data/
│ └── smartwatch_reviews.csv
│
├── models/
│ ├── naive_bayes_model.pkl
│ └── transformer_model/
│
├── notebooks/
│ ├── data_preprocessing.ipynb
│ ├── eda.ipynb
│ └── model_training.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── report/
└── Smartwatch_Sentiment_Analyzer_Report.pdf

## Project Workflow

1. **Data Collection & Cleaning**
   - Remove duplicates and missing values
   - Lowercasing, tokenization, lemmatization
   - Remove URLs, numbers, special characters

2. **Sentiment Labeling**
   - ⭐ 1–2 → Negative  
   - ⭐ 3 → Neutral  
   - ⭐ 4–5 → Positive  

3. **Feature Extraction**
   - TF-IDF (Unigrams + Bigrams)

4. **Model Development**
   - Classical ML: Naive Bayes
   - Deep Learning: BERT / RoBERTa

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix

6. **Deployment**
   - Flask backend
   - AJAX-based frontend

## Key Features

-  **Dual Model Prediction** (Classical + Transformer)
-  **Context-aware sentiment detection**
-  **Sentence-level sentiment breakdown**
-  **Real-time predictions**
-  **Confidence scores for predictions**
- **Web-based interface**

## Results & Insights

- Classical models perform well on simple and direct reviews
- Transformer models significantly outperform in:
  - Mixed sentiment reviews
  - Long reviews
  - Sarcasm and negation
- Sentence-level analysis helps identify exact strengths and weaknesses

---

##  How to Run the Project

1️. Clone the Repository

git clone https://github.com/your-username/Smartwatch-Sentiment-Analyzer.git
cd Smartwatch-Sentiment-Analyzer

2️. Install Dependencies
pip install -r requirements.txt

3️. Run the Application
python app.py

4️. Open in Browser
http://127.0.0.1:5000/

Future Enhancements:

-Multilingual sentiment analysis

-Aspect-based sentiment analysis (battery, display, fitness, etc.)

-Smartwatch-specific transformer fine-tuning

-Cloud deployment with REST API

-Real-time sentiment dashboard for businesses

References

--NLTK Documentation

--scikit-learn TF-IDF Documentation

--Hugging Face Transformers

--BERT Research Paper

--RoBERTa Research Paper
