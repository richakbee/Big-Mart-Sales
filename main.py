import json
from flask import request, Response, render_template, Flask
#from wsgiref import simple_server
from train_validate import train_validate
from train_model import train_model
from pred_validate import pred_validate
from predict_from_model import predict_from_model
from create_log_directories import create_log_directories

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def training():
    # step 0 if folder path is none then raise error else do step 1 & 2
    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            print(path)

            # step 1 validate the data
            val_obj = train_validate(path)  # object initialization
            val_obj.training_validation()  # call to method in the class
            # step2 train the model
            train_model_obj = train_model()  # object initialize
            train_model_obj.training_model()

    except ValueError:
        return Response("Error Occurred %s" % ValueError)

    except KeyError:
        return Response("Error Occurred %s" % KeyError)

    except Exception as e:
        return Response("Error Occurred %s" % e)
    return Response("Training is Successful.")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.json is not None:
            path = request.json['filepath']

            # step1 validate the data
            pred_obj = pred_validate(path)
            pred_obj.pred_validation()

            # step2 predict from model (gives path of file where prediction is saved + top 5 predictions in json
            pred_from_model_obj = predict_from_model()
            path, demo_json_predictions = pred_from_model_obj.get_prediction_from_model()

            return Response("Prediction file create at " + str(path) + ' and few of the predictions are ' + str(
                json.loads(demo_json_predictions)))

        ##if prediction request is coming from form
        elif request.form is not None:
            path = request.form['filepath']
            # step1 validate the data
            pred_obj = pred_validate(path)
            pred_obj.pred_validation()

            # step2 predict from model (gives path of file where prediction is saved + top 5 predictions in json
            pred_from_model_obj = predict_from_model()
            path, demo_json_predictions = pred_from_model_obj.get_prediction_from_model()

            return Response("Prediction file create at " + str(path) + ' and few of the predictions are ' + str(
                json.loads(demo_json_predictions)))


    except ValueError:
        return Response("Error Occurred %s" % ValueError)

    except KeyError:
        return Response("Error Occurred %s" % KeyError)

    except Exception as e:
        return Response("Error Occurred %s" % e)


if __name__ == "__main__":
    # host = '127.0.0.1'
    # port = 5000
    # #to ensure all log files are in place
    obj=create_log_directories()
    obj.create_directories()
    # server = simple_server.make_server(host, port, app)
    # server.serve_forever()
    app.run(debug=True)
