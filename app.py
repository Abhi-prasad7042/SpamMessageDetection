from flask import Flask, render_template, jsonify, request
import pickle

app = Flask(__name__)

vector = pickle.load(open("Count-vector.pkl","rb"))
model = pickle.load(open("spam-message-detection.pkl", "rb"))

@app.route('/')
def index():
    return render_template("home.html")


@app.route('/predict', methods=["POST"])
def predict():
    mess = request.form['message']
    message = vector.transform([mess])
    output = model.predict(message)[0]
    return render_template("predict.html", output = output)


if __name__ == "__main__":
    app.run(debug=True)
