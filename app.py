from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and other required objects
model = joblib.load('models/model.pkl')
le = joblib.load('models/le.pkl')
df_cols = pd.read_csv('datasets/cols.csv', header=None)
cols = df_cols.iloc[0].str.replace(' ', '').str.lower().tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        symptoms = [symptom.strip().lower() for symptom in symptoms]

        new_row = pd.Series([0]*len(cols), index=cols)
        for symptom in symptoms:
            if symptom in cols:
                new_row[symptom] = 1

        df_encoded = pd.DataFrame(columns=cols)
        df_encoded = df_encoded._append(new_row, ignore_index=True)
        
        probabilities = model.predict_proba(df_encoded)[0]
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        
        for index in top5_indices:
            disease_prob = (le.classes_[index], probabilities[index]*100)
            results.append(disease_prob)

    return render_template('index.html', symptoms=cols, results=results)

if __name__ == '__main__':
    app.run(debug=True)
