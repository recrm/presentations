import flask
import joblib
import numpy as np
import waitress


def dict_to_numpy(data):
    # Note how are keys list is hard coded. This will be important later.
    keys = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]

    return np.array([data[key] for key in keys]).reshape(1, -1) # scikit learn is very particular, it wants a matrix not an array. So we have to resize it.


def numpy_to_dict(array):
    return {
        "prediction": "virginica" if int(array[0]) == 1 else "not virginica",
    }


class ModelManager:
    def __init__(self, path_to_model):
        self.model = joblib.load(path_to_model)

    def score(self, data):
        return self.model.predict(data)


# Instantiate some neccessary global variables
app = flask.Flask(__name__) # our actual flask app
model_manager = None # This will become our model manager, but we don't want to initiate it until we actually activate our server.


# Assign a route to a specific function.
@app.route("/score", methods=["POST"])
def api_endpoint():
    data = flask.request.json
    array = dict_to_numpy(data)
    result = model_manager.score(array)

    return numpy_to_dict(result)


if __name__ == "__main__":
    # Code in here will only run if this is run as a script. Not imported.

    # Ok now we want to instantiate the model manager.
    model_manager = ModelManager("model.pkl")

    # Now we start our server
    waitress.serve(app, host="0.0.0.0", port=8080)

    # FYI, code down here will never run. That function above won't exit until forced to.
