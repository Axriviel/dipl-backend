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
import controllers.runner

from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_cors import CORS

from utils.model_storing import save_model, load_model

from controllers.model_controller import model_bp
from controllers.user_controller import user_bp
from controllers.FeedbackController import feedback_bp
from controllers.NotificationController import notification_bp


app = Flask(__name__)
#CORS(app, supports_credentials=True)  # Allow cors
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})
app.config.from_object(Config)

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


@app.route('/test', methods=['GET'])
def get_users():
    users = Config.user_service.getAll()
    if users is None:
        app.logger.error("No users found!")
        return jsonify({'error': 'No users found'}), 404

    app.logger.info(f"Found users: {users}")
    return jsonify([{'id': user.id, 'username': user.username} for user in users])    #return users

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    test = data.get("test")
    
    # Zpracování dat podle potřeby
    print(f"Received data - Name: {name}, Email: {email}, Test: {test}")
    
    response = {
        'status': 'success',
        'data': data
    }
    return jsonify(response)

# Endpoint pro zpracování požadavků
@app.route('/', methods=['GET'])
async def calculate():
    user_id = str(5)
    results = []
    if user_id in user_results:
        return jsonify({"status" : "You already have a task running"}), 409
    else:    
        task = asyncio.create_task(controllers.runner.run_async_task("xx", "xd"))
        user_results[user_id] = task

    return jsonify({'user_id': user_id}), 202

@app.route('/result/<user_id>', methods=['GET'])
async def get_result(user_id):
    if user_id in user_results:
        task = user_results[user_id]

        if task.done():
            result = task.result()
            del user_results[user_id]  # odstraníme výsledek po vrácení
            return jsonify({'result': result})
        else:
            return jsonify({'status': 'running'}), 202  # vrátíme status, že výpočet stále běží
    else:
        return jsonify({'error': 'Výsledek pro zadané ID nenalezen'}), 404

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
