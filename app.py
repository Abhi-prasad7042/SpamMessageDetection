from flask import Flask, render_template, jsonify, request
import numpy as np
import string
import nltk
from nltk.stem import PorterStemmer
import pickle

nltk.download('stopwords')
ps = PorterStemmer()

def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    word = [ps.stem(word) for word in clean_mess]
    word = " ".join(word)
    return word

app = Flask(__name__)
model = pickle.load(open("spam-message-detection.pkl", "rb"))

@app.route('/')
def index():
    return render_template("home.html")


@app.route('/predict', methods=["POST"])
def predict():
    mess = request.form.values()
    output = model.predict([mess])[0]
    return render_template("predict.html", output = output)


if __name__ == "__main__":
    app.run(debug=True)
