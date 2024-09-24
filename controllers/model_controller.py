# blueprints/model_routes.py
from flask import Blueprint, jsonify, request
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Dense
from utils.model_storing import save_model, load_model
from flask import send_file
import os

model_bp = Blueprint('model_bp', __name__)

def create_model(layers):
    model = Sequential()
    for layer in layers:
        if layer['type'] == 'Dense':
            model.add(Dense(
                units=layer['units'],
                activation=layer['activation'],
                input_shape=layer.get('input_shape', None)
            ))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_auto_model(dataset, task, user_name):
    return "ahoj"

#create task make model
@model_bp.route('/api/save-model', methods=['POST'])
def make_model():
    data = request.get_json()

    if not data or 'layers' not in data:
        return jsonify({"error": "Invalid data format"}), 400

    try:
        layers = data['layers']
        user_name = data["user"]
        print("Username:" + user_name)
        model = create_model(layers)
        save_model(model, "userModels/" + "model.keras")
        loaded_model = load_model("userModels/" + "model.keras")

        loaded_model.summary()
        #save_notification()
        return jsonify({"message": "Model successfully created and saved!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
#create task make model
@model_bp.route('/api/models/save-auto-model', methods=['POST'])
def make_auto_model():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid data format"}), 400

    try:
        dataset = data["dataset"]
        task = data['taskType']
        user_name = data["user"]
        print(dataset)
        print(task)
        print("Username:" + user_name)

        model = create_auto_model(dataset, task, user_name)

        #save_model(model, "userModels/" + "model.keras")
        #loaded_model = load_model("userModels/" + "model.keras")

        #loaded_model.summary()

        return jsonify({"message": "Model successfully created and saved!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#download selected model
@model_bp.route('/api/download-model/<int:model_id>', methods=['GET'])
def download_model(model_id):
    try:
        # Dynamická cesta k modelu na základě model_id
        model_path = f"userModels/model_{model_id}.keras"
        
        # Ověření, zda soubor existuje
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found"}), 404

        return send_file(model_path, as_attachment=True, download_name=f'model_{model_id}.keras')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#delete selected model
@model_bp.route('/api/delete-model/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    try:
        model_path = f"userModels/model_{model_id}.keras"  # cesta k modelu
        print(model_path)

        # Ověření, zda soubor existuje
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found"}), 404

        # Smazání souboru
        os.remove(model_path)
        return jsonify({"message": "Model successfully deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500