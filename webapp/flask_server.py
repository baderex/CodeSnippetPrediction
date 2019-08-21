from flask import Flask, request, render_template, redirect, url_for
from wtforms import Form, TextField,TextAreaField
import sys
import flask_ml
from sklearn.externals import joblib

app = Flask(__name__)
app.debug=True

classifier = joblib.load('classifier.pkl')
tfidfVectorizer = joblib.load('tfidfVectorizer.pkl')	

class InputForm(Form):
	code_snippet = TextAreaField('Code Snippet:')

@app.route("/home", methods=['GET','POST'])
def home():
    form = InputForm(request.form)
    if request.method == 'POST':
        predictions_dict = flask_ml.predict( form.code_snippet.data,classifier,tfidfVectorizer)
        print(predictions_dict, file=sys.stderr)
        return render_template('result.html', predictions_dict=predictions_dict)

    return render_template('home.html', form=form)

if __name__ == "__main__":
    app.run()
