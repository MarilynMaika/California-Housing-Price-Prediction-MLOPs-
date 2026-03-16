# California Housing Price Prediction (KNN Regression)

## Project Overview

This project builds a machine learning regression model to predict **median house values** using the California Housing dataset. The workflow demonstrates key **Machine Learning Operations (MLOps)** concepts including data preprocessing, pipeline creation, hyperparameter tuning, model evaluation, and saving the trained model for deployment.

## Dataset

The dataset is obtained using `fetch_california_housing` from **scikit-learn**.
The target variable represents **median house value (in $100,000s)**.

### Features

* **MedInc** – Median income in block group
* **HouseAge** – Median house age in block group
* **AveRooms** – Average number of rooms per household
* **AveBedrms** – Average number of bedrooms per household
* **Population** – Block group population
* **AveOccup** – Average number of household members
* **Latitude** – Block group latitude
* **Longitude** – Block group longitude

## Project Workflow

1. **Load Dataset**
   The California Housing dataset is loaded using `fetch_california_housing`.

2. **Train-Test Split**
   The dataset is split into **80% training data and 20% testing data**.

3. **Data Preprocessing**
   A preprocessing pipeline is created using:

   * `SimpleImputer` for handling missing values
   * `StandardScaler` for feature scaling

4. **Pipeline Construction**
   A **Pipeline** combines preprocessing and the machine learning model to ensure consistent data transformation during training and prediction.

5. **Model Training**
   A **K-Nearest Neighbors Regressor (KNN)** is used to predict housing prices.

6. **Hyperparameter Tuning**
   `GridSearchCV` with **5-fold cross-validation** is used to find the best model parameters.

   Tuned parameters:

   * `n_neighbors`
   * `weights`
   * `p` (distance metric)

7. **Model Evaluation**
   The model is evaluated using:

   * **R² Score**
   * **Mean Squared Error (MSE)**
   * **Root Mean Squared Error (RMSE)**

8. **Model Saving**
   The final trained pipeline is saved using **pickle** so it can be reused or deployed later.

## Model Performance

* **Best Parameters**

```
{'knn__n_neighbors': 9, 'knn__p': 1, 'knn__weights': 'distance'}
```

* **Best Cross-Validation R² Score:** 0.73
* **Test R² Score:** 0.72
* **Test MSE:** 0.36
* **Test RMSE:** 0.60

The model explains approximately **72% of the variance in housing prices**, with an average prediction error of about **$60,000**.

## Technologies Used

* Python
* pandas
* scikit-learn
* pickle

## Output

The trained machine learning pipeline is saved as:

```
california_knn_pipeline.pkl
```

This file contains both the **preprocessing steps and trained model**, making it ready for future predictions or deployment in an application.

## How to Run

Install dependencies using pip install

Run the app:

streamlit run streamlit_app.py


## Key Learning Outcomes

* Building machine learning pipelines
* Applying feature scaling and preprocessing
* Hyperparameter tuning using GridSearchCV
* Evaluating regression models
* Saving models for reuse and deployment

---
