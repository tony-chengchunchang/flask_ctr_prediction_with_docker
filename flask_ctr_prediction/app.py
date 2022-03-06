from flask import Flask
from ctr_model.prediction import predict_route

def create_app():
    app = Flask(__name__)
    app.register_blueprint(predict_route)

    return app
