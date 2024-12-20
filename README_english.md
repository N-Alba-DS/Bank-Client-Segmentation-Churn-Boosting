# Customer Segmentation and Banking Churn Prediction with LightGBM

This repository documents the development of a banking churn prediction model, carried out as a final assignment for the Kaggle competition organized by the subject of Data Mining in Economics and Finance of the “Master’s in Data Exploitation and Knowledge Discovery” at UBA.

[Kaggle Competition - DMEyF 2024 Third](https://www.kaggle.com/competitions/dm-ey-f-2024-tercera/leaderboard)

## Work Description

The project implements techniques for **dimensionality reduction**, **clustering**, **feature engineering**, and **boosting** (LightGBM) to predict which customers will churn within two months, taking as the test month September 2021.

First, the target variable ('clase_ternaria') was created by modifying the original dataset. This is a categorical variable, and given that the possible classes are customers who remain in the bank after two months for a given observation ("CONTINUA"), customers who churn after one month ("BAJA+1"), and customers who churn after two months ("BAJA+2"), the problem to be tackled by the model is a multi-class classification problem. In the particular case of the competition, the month to be predicted was *September 2021*.

The project was implemented using Python language in multiple Jupyter Notebook (.ipynb) scripts, with the aim that each script would solve a specific problem of the model or the Kaggle competition. Consequently, the scripts used were:

- **Recurrentes** and **Funciones**: General scripts that will be called by the script used through the magic command %run. These will be incorporated into all subsequent scripts to facilitate code articulation and readability.
- **Cluster_AE**: Used for exploratory analysis. First, the number of observations of the negative class ("CONTINUA") is subsampled. This is done because there is a significant class imbalance, aiming to avoid that the majority class observations dominate the structure of the random forest, making it more difficult to identify patterns in the minority classes: "BAJA+1" and "BAJA+2". Subsequently, a random forest is trained and then a distance matrix between samples is constructed, based on how frequently they fall into the same leaves of the set of trees. This way, a measure of dissimilarity between the samples is obtained without relying directly on Euclidean distance, but rather on the structure created by the random forest. Then UMAP is used for dimensionality reduction to two dimensions, and DBSCAN is applied to cluster and segment the customers. This strategy is employed considering the complexity of the relationships between variables and the lack of robustness of some clustering algorithms like KNN in the face of non-linear relationships in the data.
- **Back Testing**: Analyzes the real performance of the model focusing on detecting, in the month of July, the number of BAJA+2 customers.
- **Data Quality**: Allowed observing the number of nulls in the variables and performing exploratory analysis on the dataset. It uses PSI (Population Stability Index) to detect feature drifting (data drift) in the dataset.
- **Feature Engineering (FE)**: Creates lag and delta variables in the dataset. That is, values for a certain variable 1 and 2 months before the observation, and values resulting from subtracting the values 2 months before from those 1 month before, and subtracting those of 1 month before from the current ones to visualize the trend of that particular customer.
- **Punto de Corte**: Seeks to analyze through the central limit theorem what would be the ideal number of stimuli to send to the predictions made by the model. In other words, which predictions would be considered as 1 and consequently would be sent to Kaggle as the model’s final prediction.
- **VM Optuna**: In charge of optimization. Due to the size of the dataset after performing FE, a Google Virtual Machine with a configuration of 256GB of RAM and 24 virtual cores (12 real cores) was used for Bayesian hyperparameter optimization of the algorithm with the Optuna library.
- **Final Prediction**: Script used to perform a voting ensemble that would average the final prediction from a Pandas DataFrame built from the training and prediction of the model but changing multiple seeds of the model that incorporates the hyperparameters obtained by VM Optuna. This was done to control the variance of the prediction probabilities so that the scores obtained were not merely a product of chance associated with the seed used.

## Techniques Used

- **Data Preprocessing and Manipulation**:
  - Data cleaning and preparation with Pandas.
  - Concatenation of different models and datasets.
  - Subsampling.

- **Categorical Variable Encoding**:
  - **One-Hot Encoding**.
  - **Ordinal Encoding**.

- **Feature Engineering (FE)**:
  - Creation of new relevant variables (LAGS and DELTAS).
  - Transformation and rescaling of the target variable ('clase_ternaria').

- **Model Validation and Optimization**:
  - Cross-validation with **KFold** using the .cv function of lightgbm.
  - Search for optimal hyperparameters with **Optuna**.

## Conclusions

### Customer Segmentation

- **Common Characteristics**: Customers who are about to churn share a common feature: they show less activity compared to customers who remain. They can be subdivided into 3 groups with specific characteristics:

  - **Group 1**:
    - Characterized by low engagement with the services provided by the company throughout their relationship with the bank.
    - They mainly use services related to debit and the mobile application, that is, they:
      - In short, make cash withdrawals.
      - Pay for services.
      - Occasionally handle check operations.
    - They make low use of credit cards at the time of unsubscribing.
    - They progressively prepare their exit from the bank in advance, accompanying a gradual decrease in salaries deposited by their employers.
    - It is inferred that this profile corresponds to the average worker who:
      - Receives salary payment in the entity.
      - Leaves the money deposited.
      - Changes banks when changing jobs.

  - **Group 2**:
    - They use the bank to receive their salary and immediately debit it entirely.
    - In the case of unregistered work, they receive their salary in cash and only a minimal part is deposited in the bank.

  - **Group 3**:
    - In contrast to "low-engagement" (Group 1) and "low-revenue" (Group 2) customers, this segment represents **high-risk** customers.
    - Main characteristics:
      - At the time of unsubscribing, the account balance fluctuates around 0 or in slightly negative values for the majority.
      - However, a significant portion has an average debtor balance in credits and negative accounts 5 to 10 times higher than Groups 1 and 2.
    - Their activity is centered on the bank’s credit services, with sparse use of:
      - Check issuance.
      - Service payments in occasional cases.
    - Most delinquent customers at the time of unsubscribing fall into this profile:
      - Many do not complete the minimum payment on their credit cards in the months before unsubscribing.
      - It is common for them to start closing credit cards, though this is not exclusive.
      - The minimum payment amounts to avoid delinquency are on average double those of customers who continue.
    - The more advanced the account is in delinquency or closure, the higher the probability that the customer will churn.

### Prediction Model

- LightGBM turns out to be a useful algorithm to handle variables with null values such that performing imputations with techniques like Mice, Knn, or linear interpolation can worsen its performance.
- The creation of variables through feature engineering greatly contributed to improving the model’s performance.
- Performing back-testing is essential to measure the model’s performance on unseen data.
- Hyperparameter optimization through Optuna allows improving and measuring the model’s true performance.
- Using a voting ensemble derived from multiple seeds in the model for training and prediction is indispensable to deal with randomness.

The script was developed in Python and uses the following **main libraries**:

**Visualization**:
  - Plotly
  - Seaborn
  - Matplotlib

**Model**:
  - Pandas
  - Numpy
  - Lightgbm
  - Sklearn
  - Openpyxl
  - Optuna
  - Datetime
  - Imblearn.undersampling

**Exploratory Analysis**
  - DBSCAN
  - UMAP
