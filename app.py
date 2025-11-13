from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.utils import get_data_as_dataframe

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    try:
        data = get_data_as_dataframe(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )

        predict_pipe = PredictionPipeline()
        preds = predict_pipe.predict(data)
        
        result = round(preds[0], 2)
        
        return render_template('result.html', final_result=result)
    
    except ValueError as e:
        return render_template('form.html', error="Please enter valid numeric values for all fields.")
    except Exception as e:
        return render_template('form.html', error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)