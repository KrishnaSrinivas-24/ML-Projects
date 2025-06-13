# Task 4 - Movie Recommendation System using SVD (MovieLens 100k)

## 📌 Overview
This project builds a movie recommendation system using the **Singular Value Decomposition (SVD)** algorithm from the `scikit-surprise` library on the **MovieLens 100k** dataset.

## 🧠 Algorithms Used
- **SVD (Singular Value Decomposition)**: Matrix factorization technique to predict user ratings for movies.
- Evaluation with:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)

## 📊 Dataset
- **MovieLens 100k**
  - 100,000 ratings
  - 943 users
  - 1,682 movies
  - Available via `surprise.Dataset.load_builtin('ml-100k')`

## 🛠️ Features
- Trains an SVD model on the MovieLens 100k dataset.
- Evaluates model using train-test split and cross-validation.
- Calculates RMSE and MAE.
- Generates Top-N movie recommendations for each user.

## 📁 Files Included
- `Task_4.ipynb` – Jupyter Notebook with full pipeline:
  - Setup
  - Training
  - Evaluation
  - Recommendation Generation
- `README.md`

## 🚀 How to Run
1. Install required packages:
   ```bash
   pip install numpy==1.23.5
   pip install scikit-surprise --no-binary :all:
 Restart the runtime manually (especially for Colab).

 Run the rest of the notebook.

## 📦 Requirements
 Python

 NumPy (1.23.5 specifically)

 scikit-surprise

 pandas

## 🧊 Notes
 Be sure to restart the kernel after installing dependencies in Colab due to package compatibility.

 Recommendations are generated using top-N filtering with a prediction threshold.