import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras

from flask import Flask, jsonify, request, session
import asyncio
import uuid

from config.settings import Config
from models.user import db, User
import controllers.runner

from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Povolit CORS s podporou cookies
app.config.from_object(Config)

db.init_app(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

user_results = {}

# Vytvoření databázové tabulky
with app.app_context():
    db.create_all()

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password, password):
        session['user_id'] = user.id
        return jsonify({'message': 'Logged in successfully'}), 200

    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout', methods=['DELETE'])
def logout():
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/user', methods=['GET'])
def get_user():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return jsonify({'username': user.username}), 200
    return jsonify({'error': 'Not authenticated'}), 401


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
