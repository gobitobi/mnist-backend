from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from server_helpers import *
# from model.model import MyModel
from model.my_model import create_model, load_and_preprocess_data

app = Flask(__name__)
CORS(app)

HEADERS = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, PATCH, DELETE",
            "Access-Control-Allow-Headers": "Origin, X-Requested-With, Content-Type, Accept"
        }


# A simple route to check if the server is running
@app.route('/')
def home():
    # model = MyModel()
    
    # # Loads the weights
    # checkpoint_path = "training_1/cp.ckpt.weights.h5"
    # model.load_weights(checkpoint_path)

    # loss, acc = model.evaluate(model.X_test, model.y_test, verbose=2)
    # # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    return "Welcome to the Flask server!"

def find_prediction(pred_list): # find better way later
            a = [(i, item) for i, item in enumerate(pred_list)]
            max_num = 0
            max_pred = 0
            
            for i, pred in enumerate(a):
                if pred[1] > max_pred:
                    max_num = pred[0]
                    max_pred = pred[1]
            
            return max_num

# A route to handle GET requests
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        req = request.json
        data = req['data']
        # data = unflatten(data, 28, 28)
        data = np.array(data, dtype='float32')
        data = data.reshape(-1, 28, 28, 1) / 255
        print('DATA SHAPE: ', data.shape)
        
        model = create_model()
        checkpoint_path = "training_1/cp.ckpt.weights.h5"
        # checkpoint_dir = os.path.dirname(checkpoint_path)
        model.load_weights(checkpoint_path)
        # model.load_weights("test_weights.weights.h5")

        pred = model.predict(data)
        pred_list = pred[0]
        
        pred = find_prediction(pred_list)
        print(pred)
        (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
        loss, acc = model.evaluate(X_test, y_test, verbose=2)
        print("####################Restored model, accuracy: {:5.2f}%".format(100 * acc))

        
        data = { "data": pred }
        pre_res = {
            "headers": HEADERS,
            "data": str(data)
        }
        res = jsonify(pre_res)
        # print(res.data)
        return res


if __name__ == '__main__':
    app.run(port=8000, debug=True)
