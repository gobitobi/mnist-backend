from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import numpy as np
from model.my_model import create_model, load_and_preprocess_data

app = Flask(__name__)
CORS(app)

HEADERS = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, PATCH, DELETE",
            "Access-Control-Allow-Headers": "Origin, X-Requested-With, Content-Type, Accept"
        }

@app.route('/')
def home():
    return "Welcome to the Flask server!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        # receives user input request and converts it to model usable input
        data = np.array(request.json["data"], dtype="float32").reshape(-1, 28, 28, 1) / 255
        
        model = create_model()
        checkpoint_path = "training_1/cp.ckpt.weights.h5"
        model.load_weights(checkpoint_path)
        prediction = np.argmax(model.predict(data), axis=1)[0]
        
        (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
        loss, acc = model.evaluate(X_test, y_test, verbose=2)
        print("####################Restored model, accuracy: {:5.2f}%".format(100 * acc))

        return jsonify({
            "headers": HEADERS,
            "data": str({ "prediction": prediction })
        })


if __name__ == '__main__':
    app.run(port=8000, debug=True)
