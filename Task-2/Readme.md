# 🧠 Task 2: Sentiment Analysis – Customer Review Classification

This project is part of my internship at **CodTech**. The objective was to perform **sentiment analysis** on a dataset of customer reviews using **TF-IDF vectorization** and a **Logistic Regression** model.

For this project, I worked with a dataset containing **Samsung customer reviews**. Using the text content and associated ratings, I built a binary classifier to determine whether a given review is *positive* or *negative*.

---

## 📌 Objective

Perform sentiment analysis on customer reviews using:

- **TF-IDF Vectorization** to extract features from text.
- **Logistic Regression** to classify sentiment as positive or negative.

---

## 📁 Folder Structure

ML-Project/
- Task-2/
  - Sentiment_Analysis_with_NLP.ipynb  # Jupyter notebook with code and analysis
  - samsung_customer_reviews.csv       # Dataset (Samsung customer reviews)
  - Confusion-Matrix.png               # Image of the confusion matrix
  - README.md                          # You're reading it


---

## 📊 Dataset Description

**Filename**: `samsung_customer_reviews.csv`  
Contains customer feedback for Samsung products.

**Key Columns Used**:

- `Review Text`: The actual customer review text.
- `Rating`: Numerical rating from 1 to 5 stars.

**Target Variable**:  
Sentiment derived from the `Rating` column.

- Ratings ≤ 2 → `negative`
- Ratings ≥ 4 → `positive`
- Rating = 3 → *neutral (removed)*

Mapped as:  
`negative` → `0`  
`positive` → `1`

---

## 🔧 Tools & Libraries Used

- **Python**
- **Pandas & NumPy** – Data manipulation & numerical ops
- **NLTK** – Stopword removal & lemmatization
- **Scikit-learn** – TF-IDF, Logistic Regression, metrics
- **Matplotlib & Seaborn** – Visualization (Confusion Matrix)
- **Google Colab** – Development environment

---

## 🔍 Workflow & Implementation

### 📌 1. Data Loading and Initial Cleaning
- Loaded `samsung_customer_reviews.csv`.
- Selected `Review Text` and `Rating` columns.
- Dropped rows with missing values.

### 📌 2. Sentiment Definition
- Labeled reviews:
  - `Rating ≤ 2` → `negative`
  - `Rating ≥ 4` → `positive`
  - Removed `Rating = 3` entries.
- Converted labels to binary: `0 = negative`, `1 = positive`.

### 📌 3. Text Preprocessing
Created a new column `cleaned_text`:

- Lowercased all text.
- Removed punctuation, numbers, and non-alphabetic characters.
- Removed stopwords using NLTK.
- Applied lemmatization to reduce words to base form.

### 📌 4. Feature Extraction (TF-IDF Vectorization)
- Used `TfidfVectorizer` to transform text to numerical form.
- Configured with:
  - `max_features=5000`
  - `ngram_range=(1, 2)` – Unigrams and bigrams

### 📌 5. Model Training
- Split data using `train_test_split` (80/20), stratified by target.
- Trained a `LogisticRegression` model on the training set.

'''text
Training data (X_train) shape: (638, 86)
Testing data (X_test) shape: (160, 86)
Training labels (y_train) shape: (638,)
Testing labels (y_test) shape: (160,)

### 📌 6. Model Evaluation

Used the following metrics for evaluation:

- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix** (Visualized in `Confusion-Matrix.png`)

---

### 📈 Model Performance

✅ **Logistic Regression Model**  
Trained successfully on the training data. Predictions were made on the test set.

---

### 📊 Performance Metrics

**Accuracy Score**: `0.50`

#### 📄 Classification Report:

| Sentiment Class | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| Negative (0)    |   0.52    |  0.69  |   0.59   |   84    |
| Positive (1)    |   0.46    |  0.29  |   0.35   |   76    |
| **Accuracy**    |           |        |  **0.50**|   160   |
| **Macro Avg**   |   0.49    |  0.49  |   0.47   |   160   |
| **Weighted Avg**|   0.49    |  0.50  |   0.48   |   160   |

---



### 🖼️ Confusion Matrix

![Confusion Matrix](https://github.com/KrishnaSrinivas-24/ML-Projects/blob/main/Task-2/Confusion-Matix.png)

---

### 🚀 Future Work

- Experiment with other models like **SVM**, **Random Forest**, or **XGBoost**
- Tune hyperparameters for improved performance
- Expand the dataset to include reviews from other brands/products
- Explore deep learning techniques like **LSTM** or **BERT**

> “Sentiment analysis: where every word is a clue, and every pattern reveals the truth behind the mask.”
