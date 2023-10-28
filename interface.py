# Import all necessary libraries and magics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
plt.style.use('bmh')
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, Lasso, Ridge

model = joblib.load('models/model.pkl')
le = joblib.load('models/le.pkl')

# Get symptoms from user
print('Please enter your symptoms separated by commas:')
symptoms = input().split(',')
symptoms = [symptom.strip().lower() for symptom in symptoms]  # Added lower() and strip() for consistency

# Get columns from model
df_cols = pd.read_csv('datasets/cols.csv', header=None)
cols = df_cols.iloc[0].str.replace(' ', '').str.lower().tolist()

# Create a new row filled with zeros
new_row = pd.Series([0]*len(cols), index=cols)

# Encode the symptoms into the new row
for symptom in symptoms:
    if symptom in cols:
        new_row[symptom] = 1

# Add the new row to the dataframe
df_encoded = pd.DataFrame(columns=cols)
df_encoded = df_encoded._append(new_row, ignore_index=True)

# Get the predicted probabilities and the top 5 indices
probabilities = model.predict_proba(df_encoded)[0]
top5_indices = np.argsort(probabilities)[-5:][::-1]  # Get indices of top 5 probabilities

# Print the 5 highest probabilities and their corresponding diseases
print('The 5 most likely diseases are:')
for index in top5_indices:
    print(f"{le.classes_[index]}: {probabilities[index]*100:.2f}%")
