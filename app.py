import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras

from flask import Flask, jsonify, request, session
import asyncio
import uuid

import controllers.runner
from repositories.UserRepository import UserRepository
from services.UserService import UserService

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Povolit CORS s podporou cookies
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # nebo jiná DB
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['CORS_SUPPORTS_CREDENTIALS'] = True

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
from flask_cors import CORS

# Inicializace UserRepository a UserService
user_repository = UserRepository()
user_service = UserService(user_repository)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

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

    users = user_service.getAll()
    #users_list = [{'id': user.id, 'username': user.username, 'email': user.email} for user in users]
    return jsonify(users)


# Endpoint pro zpracování požadavků
@app.route('/', methods=['GET'])
async def calculate():
    # data = request.get_json()
    # params = data['params']  # předpokládáme, že získáme parametry z requestu
    # user_id = str(uuid.uuid4())
    user_id = str(5)
    results = []

    task = asyncio.create_task(controllers.runner.run_async_task("xx", "xd"))
    user_results[user_id] = task

    return jsonify({'user_id': user_id}), 202



@app.route('/result/<user_id>', methods=['GET'])
async def get_result(user_id):
    # return(user_results)
    if user_id in user_results:
        task = user_results[user_id]

        if task.done():
            result = task.result()
            del user_results[user_id]  # odstraníme výsledek po vrácení
            print(result)
            # return result
            return jsonify({'result': result})
        else:
            return jsonify({'status': 'running'}), 202  # vrátíme status, že výpočet stále běží
    else:
        return jsonify({'error': 'Výsledek pro zadané ID nenalezen'}), 404

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)