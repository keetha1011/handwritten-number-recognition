from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import model
from model import prediction

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', prediction="")


@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    return jsonify({"Prediction": str(prediction(data))})
@app.route('/styles.css', methods=['GET'])
def styles():
    return send_file('assets/styles.css')

if __name__ == '__main__':
    app.run(debug=True)