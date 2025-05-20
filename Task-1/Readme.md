# 🧠 Task 1: Decision Tree Classifier – PUBG Weapon Type Prediction

This project is part of my internship at **CodTech**. The task was to build and visualize a **Decision Tree model** using Scikit-learn to classify or predict outcomes on a selected dataset.

For this project, I chose a dataset containing stats of weapons from the popular game **PUBG**, and used these features to predict the **weapon type**.

---

## 📌 Objective

> Build and visualize a Decision Tree model using Scikit-learn to classify or predict outcomes on a chosen dataset.

---

## 📁 Folder Structure

ML-Project/

└── Task-1/

├── Decision Tree.ipynb # Jupyter notebook with code and analysis

├── data.csv # Dataset (PUBG weapon stats)

└── README.md # You're reading it

---

## 📊 Dataset Description

The dataset used contains detailed stats of various weapons in PUBG. Some of the key features include:

- **Damage**
- **Magazine Capacity**
- **Bullet Speed**
- **Rate of Fire**
- **Shots to Kill (Chest/Head)**
- **Fire Mode**
- **Range**

The target variable is `Weapon Type`, which the model tries to predict.

---

## 🔧 Tools & Libraries Used

- **Python**
- **Pandas & NumPy** – for data manipulation
- **Matplotlib & Seaborn** – for data visualization
- **Scikit-learn** – for machine learning (decision tree, model evaluation)
- **Google Colab** – for coding and execution

---

## 🔍 Workflow & Implementation

### 📌 1. Data Preprocessing
- Uploaded the dataset via Google Colab.
- Dropped irrelevant columns:
  - `Weapon Name`, `Weapon Type`, `Bullet Type`, `Fire Mode`
- Handled missing values by filling them with 0.
- Encoded the target variable (`Weapon Type`) using `LabelEncoder`.

### 📌 2. Train-Test Split
- Split the data using `train_test_split()` into:
  - **Training set:** 80%
  - **Test set:** 20%

### 📌 3. Model Training
- Used `DecisionTreeClassifier` from `sklearn`.
- Trained the model on the training data.

### 📌 4. Model Evaluation
- Evaluated the model using:
  - `classification_report`
  - `accuracy_score`
- Observed class-wise precision, recall, and f1-score.

### 📌 5. Visualization
- Visualized the trained Decision Tree using:
  - `plot_tree()` from `sklearn.tree`
- Displayed feature names and class labels for better interpretability.

---

## 📈 Sample Output (Classification Report)
                  precision    recall   f1-score   support
Assault Rifle       0.67      1.00      0.80         2
Designed Marksman   0.33      0.50      0.40         2
Melee               1.00      1.00      1.00         1
Pistol              0.00      0.00      0.00         3
Submachine Gun      0.50      1.00      0.67         1
Accuracy                                0.56         9
Macro avg           0.50      0.70      0.57         9
Weighted avg        0.39      0.56      0.45         9


⚠️ **Note:** Some classes like "Pistol" were not predicted well. This is likely due to class imbalance or limited dataset size.

---

## 🌳 Decision Tree Visualization

The model was visualized with all feature splits and decisions shown using `plot_tree()`. The tree helps interpret:
- Which features were important
- How the decision boundaries were formed
- How classification was achieved

---

## ✅ Outcomes & Learnings

- Understood how Decision Trees classify data.
- Visualized the internal structure of a trained tree.
- Learned how class imbalance affects prediction performance.
- Learned the importance of proper preprocessing and feature selection.

---

## 📜 Internship Note

This project is part of my internship deliverables at **CodTech**.  
A certificate of completion will be issued at the end of the internship.

---

## 💡 Improvements (Future Scope)

- Use a larger and more balanced dataset.
- Try out more robust classifiers like:
  - Random Forest
  - Gradient Boosting
- Perform feature scaling or dimensionality reduction for complex data.
- Tune hyperparameters using Grid Search or Cross-Validation.

---

### ✍️ Author: Krishna  
_Passionate about machine learning, coding, and building real-world projects._

> “Don’t play the odds. Play the man.” – Harvey Specter

