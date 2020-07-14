from flask import Flask, request, render_template
import joblib

# Load model pipeline from disk
loaded_pipeline = joblib.load('../05_Models/fake_title_SVM_pipeline.sav')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def submit():

    template_data = {
        'headline' : '',
        'prediction': '',
        'percent': '',
        'prediction_color': 'black',
        'percent_color': 'black'
        }

    if request.method == 'POST':
        text = request.form['text']

        template_data['headline'] = text

        template_data['prediction'] = str(loaded_pipeline.predict([text])[0])
        if template_data['prediction'] == 'True':
            template_data['percent'] = round(loaded_pipeline.predict_proba([text])[0][1] * 100, 2)
            template_data['prediction_color'] = 'green'
        else:
            template_data['percent'] = round(loaded_pipeline.predict_proba([text])[0][0] * 100, 2)
            template_data['prediction_color'] = 'red'

        if template_data['percent'] < 60:
            template_data['percent_color'] = 'red'
        elif template_data['percent'] < 70:
            template_data['percent_color'] = 'orange'
        elif template_data['percent'] < 80:
            template_data['percent_color'] = 'yellow'
        elif template_data['percent'] <= 90:
            template_data['percent_color'] = 'green'         
        else:
            template_data['percent_color'] = 'blue'

    return render_template('form.html', **template_data)

if __name__=='__main__':
    app.run(debug=True)