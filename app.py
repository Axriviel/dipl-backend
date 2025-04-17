import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

from flask import Flask, jsonify, request
import asyncio

from config.settings import Config
from models.user import db, User
from models.model import Model
from models.feedback import Feedback
from models.notification import Notification

from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_cors import CORS

from utils.model_storing import save_model, load_model

from controllers.model_controller import model_bp
from controllers.user_controller import user_bp
from controllers.feedback_controller import feedback_bp
from controllers.notification_controller import notification_bp
from controllers.dataset_controller import dataset_bp


app = Flask(__name__)
app.config.from_object(Config)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["https://kerasage.axriviel.eu", "http://localhost:5173"]}})

db.init_app(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

user_results = {}

# create database tables if needed
with app.app_context():
    db.create_all()

# register blueprints
app.register_blueprint(user_bp)
app.register_blueprint(model_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(notification_bp)
app.register_blueprint(dataset_bp)


@app.route("/", methods=["GET"])
def is_alive():
    return jsonify({"message" : "I am alive"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
