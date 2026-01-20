# Social_Media_data_Sentiment_Analysis_Fall_2025_Proj
A machine learning–based social media sentiment analysis project that preprocesses text data, extracts features , and evaluates multiple classification models to identify public sentiment (positive, negative, neutral) from social media data.


## 📌 Project Overview
This project focuses on **sentiment analysis of social media data** using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. The goal is to automatically classify social media posts into **positive, negative, or neutral** sentiment categories based on textual content.

The project follows a complete machine learning pipeline starting from raw text data to model evaluation and comparison.

---

## 🎯 Objectives
- Analyze public sentiment from social media text data  
- Perform effective text preprocessing and feature extraction  
- Compare multiple machine learning models for sentiment classification  
- Identify the most accurate and computationally efficient model  

---

----
## 🗂️ Dataset

Due to GitHub file size limitations, the **Sentiment140 dataset** is not included directly in this repository.

You can download the dataset from the official Kaggle source:

🔗 **Sentiment140 Dataset (Kaggle)**  
https://www.kaggle.com/datasets/kazanova/sentiment140

### Dataset Description
- Contains **1.6 million Twitter tweets**
- Tweets are labeled based on sentiment polarity:
  - `0` → Negative
  - `4` → Positive
- Originally collected using the Twitter API
- Commonly used for **sentiment analysis and NLP research**

### Dataset Format
The dataset is provided as a CSV file with the following columns:
1. Target (sentiment label)
2. Tweet ID
3. Date
4. Query
5. Username
6. Tweet text

### How to Use
1. Download the dataset from Kaggle using the link above
2. Extract the CSV file
3. Place it inside the `data/` directory of this project
4. Rename it if required (e.g., `dataset.csv`) to match the code

> Note: A Kaggle account is required to download the dataset.



---

## 🧹 Text Preprocessing Steps
- Convert text to lowercase  
- Remove URLs, mentions, hashtags, and special characters  
- Handle emojis and emoticons  
- Remove stopwords  
- Lemmatization for word normalization  
- POS tagging to retain sentiment-relevant words  

---

## 🧠 Feature Extraction
- **TF-IDF Vectorization**
  - Unigrams  
  - Bigrams  
- Converts textual data into numerical form for ML models  

---

## 🤖 Machine Learning Models Used
- Logistic Regression ✅ (Best performing)  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Other baseline classifiers (if applicable)  

---

## 📊 Evaluation Metrics
Models are evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 🏆 Results Summary
- Hyperparameter tuning improved overall model performance  
- **Logistic Regression achieved the highest accuracy**  
- Stable performance across all evaluation metrics  
- Lower computational cost compared to other models  

**Reason for selecting Logistic Regression:**
- Efficient on high-dimensional sparse data (TF-IDF)  
- Robust and interpretable  
- Faster training and prediction time  

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Libraries:**
  - NumPy  
  - Pandas  
  - NLTK / spaCy  
  - Scikit-learn  
  - Matplotlib / Seaborn  

---

## 👤 Author
**Muhammad Zeeshan Nasrullah**  
Final Year Student | Machine Learning & NLP Enthusiast  

This project was developed as an academic machine learning project focusing on sentiment analysis of social media data using NLP techniques and supervised learning models.



---

## ▶️ How to Run the Project
1. Clone the repository:
```bash
git clone https://github.com/zeeshannasrullah/Social_Media_data_Sentiment_Analysis.git

