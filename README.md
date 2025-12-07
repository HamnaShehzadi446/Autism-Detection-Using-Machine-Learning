# Autism Spectrum Disorder (ASD) Prediction using Machine Learning

This project implements supervised machine learning algorithms to predict Autism Spectrum Disorder (ASD) in adults using demographic and behavioral assessment data. Six models were used and evaluated: Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Naive Bayes, and Logistic Regression. The models are optimized using hyperparameter tuning and evaluated with metrics such as accuracy, precision, recall, F1 score, and F-beta score.


## **Getting Started**

### **Prerequisites**

* Python 3.x
* Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
* Jupyter Notebook or Google Colab / VS Code


### **Running the Project**

#### **Option 1: Using Google Colab**

1. Upload the project files, including:

   * The main notebook (autism_detection.ipynb)
   * `autism_data.csv` (dataset)
   * `visuals.py` (visualization helper)
2. Ensure all files are in the same Colab session directory.
3. Run the notebook cells sequentially.

#### **Option 2: Using VS Code / Local Environment**

1. Place all project files in a single folder:

   * The main notebook (autism_detection.ipynb)
   * `autism_data.csv` (dataset)
   * `visuals.py` (visualization helper)
2. Open the folder in VS Code.
3. Install required libraries:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
4. Run the Python script or Jupyter Notebook.


## **Features**

* Preprocessing: Missing value handling, scaling, and one-hot encoding
* Six supervised ML models with hyperparameter tuning
* Model evaluation using accuracy, precision, recall, F1 score, and F-beta score
* Confusion matrix visualization for each model

