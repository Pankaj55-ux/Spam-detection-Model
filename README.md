# 📧 Spam Detector using Machine Learning
## 📌 Overview
This project is a machine learning-based spam detection system that classifies messages as **spam** or **ham** with **98% accuracy**.  
It uses natural language processing (NLP) techniques for text preprocessing and multiple ML algorithms for classification.
## ✨ Features
- 📊 High accuracy (98%) with robust F1 score
- 📝 Preprocessing: tokenization, stopword removal, stemming
- 🔢 Feature extraction using Bag of Words (BoW) and TF-IDF
- 🤖 Models: Logistic Regression, Naïve Bayes, SVM
- 📈 Evaluation using Precision, Recall, and F1 score
## 📂 Dataset
The dataset used is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) from UCI Machine Learning Repository.

- **Total Messages:** 5,574  
- **Labels:** Spam / Ham
## 🛠 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK
- **Jupyter Notebook** for development
## ⚙️ Workflow
1. **Load Dataset** → Read the dataset into a DataFrame.
2. **Data Preprocessing** → Tokenization, stopword removal, stemming.
3. **Feature Extraction** → Convert text into numerical vectors using BoW and TF-IDF.
4. **Model Training** → Train Logistic Regression, Naïve Bayes, and SVM models.
5. **Model Evaluation** → Calculate Accuracy, Precision, Recall, and F1 score.
6. **Comparison & Selection** → Choose the best model for final predictions.
## 📊 Model Performance
| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.98     | 0.98      | 0.97   | 0.97     |
| Naïve Bayes          | 0.97     | 0.96      | 0.95   | 0.96     |
| SVM                  | 0.98     | 0.98      | 0.97   | 0.97     |

**Best Model:** Logistic Regression & SVM (tied at 98% accuracy)
## 📑 Classification Report
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99      1205
        spam       0.99      0.94      0.96       188

    accuracy                           0.98      1393
   macro avg       0.98      0.97      0.97      1393
weighted avg       0.98      0.98      0.98      1393
