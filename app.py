from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Combined route for home page and prediction
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Retrieve and process form data
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            pred_df = data.get_data_as_data_frame()

            # Make prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=results[0])
        except Exception as e:
            return render_template('home.html', error=str(e))

    # GET request: Render the form
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
