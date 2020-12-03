#IMPORT MODULES
from flask import Flask,jsonify,request
from classifier import get_prediction

# CREATE THE APP
app = Flask(__name__)

# GIVE THE ROUTE
@app.route("/predict-alphabet",methods = ["POST"])

# CREATE THE PREDICTION
def predict_data():
    image = request.files.get("alphabets")
    prediction = get_prediction(image)
    print("Prediction is ", prediction)
    return jsonify({
        "prediction":prediction
    }),200

# RUN IT
if __name__ == "__main__":
    app.run(debug = True)