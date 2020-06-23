from flask import Flask, render_template, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import numpy as np  
#from tensorflow.keras.models import load_model
import joblib



def return_prediction(model,scaler,sample_json):
    
    Pregnancies = sample_json['Pregnancies']
    Glucose = sample_json['Glucose']
    BloodPressure = sample_json['BloodPressure']
    SkinThickness = sample_json['SkinThickness']
    Insulin = sample_json['Insulin']
    BMI = sample_json['BMI']
    DiabetesPedigreeFunction = sample_json['DiabetesPedigreeFunction']
    Age = sample_json['Age']
    
    diabetes = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
    
    diabetes = scaler.transform(diabetes)
    
    classes = np.array(['Not Diabetic!', 'Diabetic!'])
    
    class_ind = model.predict(diabetes)
    
    return classes[class_ind][0]



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
loaded_model = joblib.load('final_diab_model.h5')
loaded_scaler = joblib.load('diab_scaler.pkl')


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class DiabetesForm(FlaskForm):
    Pregnancies = TextField('Pregnancies', render_kw={"placeholder": "eg. 2", "size":20})
    Glucose = TextField('Glucose', render_kw={"placeholder": "(mg/dl) eg. 100", "size":20})
    BloodPressure = TextField('BloodPressure', render_kw={"placeholder": "(mmHg) eg. 90", "size":20})
    SkinThickness = TextField('SkinThickness', render_kw={"placeholder": "(mm) eg. 80", "size":20})
    Insulin = TextField('Insulin', render_kw={"placeholder": "(IU/mL) eg. 80", "size":20})
    BMI = TextField('BMI', render_kw={"placeholder": "(kg/m^2) eg. 26.1", "size":20})
    DiabetesPedigreeFunction = TextField('DiabetesPedigreeFunction', render_kw={"placeholder": "eg. 0.56", "size":20})
    Age = TextField('Age', render_kw={"placeholder": "(years) eg. 35", "size":20})
    
    submit = SubmitField('Analyze')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = DiabetesForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['prg'] = form.Pregnancies.data
        session['glc'] = form.Glucose.data
        session['bp'] = form.BloodPressure.data
        session['skt'] = form.SkinThickness.data
        session['ins'] = form.Insulin.data
        session['bmi'] = form.BMI.data
        session['dpf'] = form.DiabetesPedigreeFunction.data
        session['age'] = form.Age.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form = form)


@app.route('/prediction')
def prediction():

    content = {}

    content['Pregnancies'] = float(session['prg'])
    content['Glucose'] = float(session['glc'])
    content['BloodPressure'] = float(session['bp'])
    content['SkinThickness'] = float(session['skt'])
    content['Insulin'] = float(session['ins'])
    content['BMI'] = float(session['bmi'])
    content['DiabetesPedigreeFunction'] = float(session['dpf'])
    content['Age'] = float(session['age'])

    results = return_prediction(model = loaded_model, scaler = loaded_scaler, sample_json = content)
    
    if results == 'Diabetic!':
        return render_template('prediction.html',results=results)
    else:
        return render_template('prediction1.html',results=results)

if __name__ == '__main__':
    app.run(debug=True)
