from flask import Flask, request, render_template
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model pipeline from disk
title_pipeline = joblib.load('05_Models/large_title_SVM_pipeline.sav')
title_pipeline2 = joblib.load('05_Models/small_title_SVM_pipeline.sav')

body_pipeline = joblib.load('05_Models/large_body_SVM_pipeline.sav')
body_pipeline2 = joblib.load('05_Models/small_body_SVM_pipeline.sav')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('Analysis.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/visualization')
def visualization():
    return render_template('Visualization.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

    template_data = {
        'headline':'',
        'pipeline1': {
            'prediction': '',
            'percent': '',
            'prediction_color': 'black',
            'percent_color': 'black'
        },
        'pipeline2': {
            'prediction': '',
            'percent': '',
            'prediction_color': 'black',
            'percent_color': 'black'
        }
    }

    if request.method == 'POST':
        text = request.form['text']

        template_data['headline'] = text.lower()

        # First model
        template_data['pipeline1']['prediction'] = str(title_pipeline.predict([text.lower()])[0])
        if template_data['pipeline1']['prediction'] == 'True':
            template_data['pipeline1']['percent'] = round(title_pipeline.predict_proba([text.lower()])[0][1], 4)
            template_data['pipeline1']['prediction_color'] = 'green'
        else:
            template_data['pipeline1']['percent'] = round(title_pipeline.predict_proba([text.lower()])[0][0], 4)
            template_data['pipeline1']['prediction_color'] = 'red'

        if template_data['pipeline1']['percent'] < .60:
            template_data['pipeline1']['percent_color'] = 'red'
        elif template_data['pipeline1']['percent'] < .70:
            template_data['pipeline1']['percent_color'] = 'orange'
        elif template_data['pipeline1']['percent'] < .80:
            template_data['pipeline1']['percent_color'] = 'yellow'
        elif template_data['pipeline1']['percent'] <= .90:
            template_data['pipeline1']['percent_color'] = 'green'         
        else:
            template_data['pipeline1']['percent_color'] = 'blue'

        # Second model
        template_data['pipeline2']['prediction'] = str(title_pipeline2.predict([text])[0])
        if template_data['pipeline2']['prediction'] == 'True':
            template_data['pipeline2']['percent'] = round(title_pipeline2.predict_proba([text])[0][1], 4)
            template_data['pipeline2']['prediction_color'] = 'green'
        else:
            template_data['pipeline2']['percent'] = round(title_pipeline2.predict_proba([text])[0][0], 4)
            template_data['pipeline2']['prediction_color'] = 'red'

        if template_data['pipeline2']['percent'] < .60:
            template_data['pipeline2']['percent_color'] = 'red'
        elif template_data['pipeline2']['percent'] < .70:
            template_data['pipeline2']['percent_color'] = 'orange'
        elif template_data['pipeline2']['percent'] < .80:
            template_data['pipeline2']['percent_color'] = 'yellow'
        elif template_data['pipeline2']['percent'] <= .90:
            template_data['pipeline2']['percent_color'] = 'green'         
        else:
            template_data['pipeline2']['percent_color'] = 'blue'

    return render_template('Prediction.html', **template_data)

@app.route('/predict_body', methods=['GET', 'POST'])
def predict_body():

    template_data = {
        'headline':'',
        'pipeline1': {
            'prediction': '',
            'percent': '',
            'prediction_color': 'black',
            'percent_color': 'black'
        },
        'pipeline2': {
            'prediction': '',
            'percent': '',
            'prediction_color': 'black',
            'percent_color': 'black'
        }
    }


    if request.method == 'POST':
        text = request.form['text']

        template_data['headline'] = text

        # First model
        template_data['pipeline1']['prediction'] = str(body_pipeline.predict([text.lower()])[0])
        if template_data['pipeline1']['prediction'] == 'True':
            template_data['pipeline1']['percent'] = round(body_pipeline.predict_proba([text.lower()])[0][1], 4)
            template_data['pipeline1']['prediction_color'] = 'green'
        else:
            template_data['pipeline1']['percent'] = round(body_pipeline.predict_proba([text.lower()])[0][0], 4)
            template_data['pipeline1']['prediction_color'] = 'red'

        if template_data['pipeline1']['percent'] < .60:
            template_data['pipeline1']['percent_color'] = 'red'
        elif template_data['pipeline1']['percent'] < .70:
            template_data['pipeline1']['percent_color'] = 'orange'
        elif template_data['pipeline1']['percent'] < .80:
            template_data['pipeline1']['percent_color'] = 'yellow'
        elif template_data['pipeline1']['percent'] <= .90:
            template_data['pipeline1']['percent_color'] = 'green'         
        else:
            template_data['pipeline1']['percent_color'] = 'blue'

        
        # Second model
        template_data['pipeline2']['prediction'] = str(body_pipeline2.predict([text])[0])
        if template_data['pipeline2']['prediction'] == 'True':
            template_data['pipeline2']['percent'] = round(body_pipeline2.predict_proba([text])[0][1], 4)
            template_data['pipeline2']['prediction_color'] = 'green'
        else:
            template_data['pipeline2']['percent'] = round(body_pipeline2.predict_proba([text])[0][0], 4)
            template_data['pipeline2']['prediction_color'] = 'red'

        if template_data['pipeline2']['percent'] < .60:
            template_data['pipeline2']['percent_color'] = 'red'
        elif template_data['pipeline2']['percent'] < .70:
            template_data['pipeline2']['percent_color'] = 'orange'
        elif template_data['pipeline2']['percent'] < .80:
            template_data['pipeline2']['percent_color'] = 'yellow'
        elif template_data['pipeline2']['percent'] <= .90:
            template_data['pipeline2']['percent_color'] = 'green'         
        else:
            template_data['pipeline2']['percent_color'] = 'blue'

    return render_template('Prediction2.html', **template_data)

@app.route('/facehug_baseline', methods=['GET', 'POST'])
def facehug():
    tokenizer = AutoTokenizer.from_pretrained("ghanashyamvtatti/roberta-fake-news")
    model = AutoModelForSequenceClassification.from_pretrained("ghanashyamvtatti/roberta-fake-news")
    template_data = {'text':'', 'pred':'', 'prob':'', 'prediction_color':'', 'percent_color':''}
    if request.method == 'POST':
        text = request.form['text']

        inputs = tokenizer(text, return_tensors='pt')
        output = model(**inputs)

        pred_logits = output.logits.softmax(dim=1).detach().cpu().flatten().numpy().tolist()
        prob = round(max(pred_logits), 2)
        template_data['prob'] = prob
        pred = pred_logits.index(prob)

        if pred == 0:
            template_data['pred'] = 'False'
            template_data['prediction_color'] = 'Red'
        else:
            template_data['pred'] = 'True'
            template_data['prediction_color'] = 'Green'

        if template_data['prob'] < .60:
            template_data['percent_color'] = 'red'
        elif template_data['prob'] < .70:
            template_data['percent_color'] = 'orange'
        elif template_data['prob'] < .80:
            template_data['percent_color'] = 'yellow'
        elif template_data['prob'] <= .90:
            template_data['percent_color'] = 'green'         
        else:
            template_data['percent_color'] = 'blue'

    return render_template('facehug.html', **template_data)

if __name__=='__main__':
    app.run(debug=True)
