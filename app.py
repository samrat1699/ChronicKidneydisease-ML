from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('final_forest_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        sg = float(request.form['specific_gravity'])
        htn = float(request.form['hypertension'])
        hemo = float(request.form['haemoglobin'])
        dm = float(request.form['diabetes_mellitus'])
        al = float(request.form['albumin'])
        #appet = float(request.form['appet'])
        rc = float(request.form['red_blood_cell_count'])
        #sc = float(request.form['serum_creatinine'])
        pc = float(request.form['packed_cell_volume'])
        #bgr = float(request.form['blood_glucose_random'])
        #sod = float(request.form['sodium'])
        values = np.array([[sg, htn, hemo, dm, al, rc, pc]]).astype(float)
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
