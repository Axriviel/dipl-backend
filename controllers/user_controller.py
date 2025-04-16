from flask import Blueprint, jsonify, request, session
from models.user import db, User
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()
user_bp = Blueprint('user_bp', __name__)

registrations_allowed = True

# API key validation
def validate_api_key():
    req_api_key = request.headers.get('X-API-KEY')
    from config.settings import Config
    return req_api_key == Config.API_KEY

#registration
@user_bp.route('/api/user/register', methods=['POST'])
def register():
    if not registrations_allowed:
        return jsonify({'error': 'Registrations not allowed atm'}), 403
    
    if not validate_api_key():
        return jsonify({'error': 'Unauthorized request'}), 403

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400

    #hash passwd
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201

#login
@user_bp.route('/api/user/login', methods=['POST'])
def login():
    if not validate_api_key():
        return jsonify({'error': 'Unauthorized request'}), 403

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password, password):
        session["user_id"] = user.id
        return jsonify({'message': 'Logged in successfully'}), 200

    return jsonify({'error': 'Invalid credentials'}), 401

#logout
@user_bp.route('/api/user/logout', methods=['DELETE'])
def logout():
    user_id = session.get('user_id')  

    if user_id is None:
        return jsonify({'message': 'User is not logged in'}), 400

    # Session cleanup
    session.pop('user_id', None) 
    return jsonify({'message': 'Logged out successfully'}), 200

#get current user
@user_bp.route('/api/user/getUser', methods=['GET'])
def get_user():
    user_id = session.get('user_id') 

    if user_id:
        user = User.query.get(user_id)
        if user:
            return jsonify({"id": user.id, 'username': user.username}), 200

    return jsonify({'error': 'Not authenticated'}), 401
