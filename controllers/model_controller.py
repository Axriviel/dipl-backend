# blueprints/model_routes.py
from flask import Blueprint, jsonify, request
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Dense
from utils.model_storing import save_model, load_model
from flask import send_file
from controllers.notification_controller import create_notification
from controllers.user_controller import session
from models.model import Model, db
import random
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
        
        model = create_model(layers)
        save_model(model, "userModels/" + "model.keras")
        loaded_model = load_model("userModels/" + "model.keras")

        loaded_model.summary()
        create_notification(for_user_id = session.get("user_id"), message = "Model created")
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
        user_id = session.get('user_id')  # Získání user_id ze session
        dataset = data["dataset"]
        task = data['taskType']
        opt_method = data["optMethod"]
        print(dataset)
        print(task)
        print("Opt method: " + opt_method)

        model = create_auto_model(dataset, task, user_id)

        #přesunout do funkce
        new_model = Model(model_name = "model_"+ str(random.random()), accuracy = 0.75, error = 0.07, dataset = dataset, user_id = user_id)
        db.session.add(new_model)
        db.session.commit()

        create_notification(for_user_id = user_id, message = "Model creating")


        #save_model(model, "userModels/" + "model.keras")
        #loaded_model = load_model("userModels/" + "model.keras")

        #loaded_model.summary()

        return jsonify({"message": "Model successfully created and saved!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

#get all models
@model_bp.route("/api/getmodels", methods=['GET'])
def return_models():
    try:
        data = Model.query.all()
                # Serializace dat do seznamu slovníků
        model_list = [
            {
                'id': model.id,
                'name': model.model_name,
                "accuracy": model.accuracy,
                "error": model.error,
                "dataset": model.dataset,

            } 
            for model in data
        ]
        
        # Vrácení dat jako JSON odpověď
        return jsonify(model_list)
    except Exception as e:
        print(str(e))
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