from flask import Flask, render_template, request
import numpy as np
import pickle
import sqlite3
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('final_forest_model.pkl', 'rb'))

# Ensure the folder exists
db_folder = 'predictiondb'
if not os.path.exists(db_folder):
    os.makedirs(db_folder)

# Path to the SQLite database
db_path = os.path.join(db_folder, 'predictions.db')

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create predictions table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        specific_gravity REAL,
        hypertension INTEGER,
        haemoglobin REAL,
        diabetes_mellitus INTEGER,
        albumin INTEGER,
        red_blood_cell_count REAL,
        packed_cell_volume INTEGER,
        prediction INTEGER
    )
    ''')
    conn.commit()
    conn.close()

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Fetch form inputs
        sg = float(request.form['specific_gravity'])
        htn = float(request.form['hypertension'])
        hemo = float(request.form['haemoglobin'])
        dm = float(request.form['diabetes_mellitus'])
        al = float(request.form['albumin'])
        rc = float(request.form['red_blood_cell_count'])
        pc = float(request.form['packed_cell_volume'])

        # Prepare the input data for prediction
        values = np.array([[sg, htn, hemo, dm, al, rc, pc]]).astype(float)
        prediction = model.predict(values)[0]

        # Save the prediction to the database
        save_prediction(sg, htn, hemo, dm, al, rc, pc, prediction)

        return render_template('result.html', prediction=prediction)

def save_prediction(sg, htn, hemo, dm, al, rc, pc, prediction):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO predictions (specific_gravity, hypertension, haemoglobin, diabetes_mellitus, albumin,
                            red_blood_cell_count, packed_cell_volume, prediction)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (sg, htn, hemo, dm, al, rc, pc, prediction))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Initialize the database (create tables if needed)
    init_db()

    # Run the Flask app
    app.run(debug=True)
