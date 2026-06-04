from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/predict', methods=['POST'])
def predict():

    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    area = float(request.form['area'])

    features = np.array([[bedrooms, bathrooms, area]])

    prediction = model.predict(features)

    return render_template(
        'result.html',
        prediction=round(prediction[0], 2)
    )

    return render_template(
        'result.html',
        predicted_price=price
    )

if __name__ == '__main__':
    app.run(debug=True)