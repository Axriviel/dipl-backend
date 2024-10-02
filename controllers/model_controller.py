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
import time


model_bp = Blueprint('model_bp', __name__)

active_tasks = {}

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
    model = Sequential()
    model.add(Dense(8))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#create task make model
@model_bp.route('/api/save-model', methods=['POST'])
def make_model():
    data = request.get_json()
    user_id = session.get('user_id')

    if not data or 'layers' not in data:
        return jsonify({"error": "Invalid data format"}), 400
    
        #user already has running task
    if user_id in active_tasks:
        return jsonify({"error": "Task already running. Please wait until it finishes."}), 400

    try:
        active_tasks[user_id] = True

        layers = data['layers']
        model = create_model(layers)

        #save_model(model, "userModels/" + "model.keras")
        #loaded_model = load_model("userModels/" + "model.keras")

        #loaded_model.summary()

        # new_model = Model(model_name = "model_"+ str(round(random.random(), 3)), accuracy = 0.75, error = 0.07, dataset = "test", user_id = user_id)
        # db.session.add(new_model)
        # db.session.commit()
        # model_id = new_model.id

        # save_model(model, user_id, model_id)
        # create_notification(for_user_id = session.get("user_id"), message = "Model created")

        dataset = "test"
        save_and_notification(model, user_id, dataset)
        
        #task finished, remove user from active
        active_tasks.pop(user_id, None)
        return jsonify({"message": "Model successfully created and saved!"}), 200
    except Exception as e:
        active_tasks.pop(user_id, None)
        return jsonify({"error": str(e)}), 500
    
#create task make model
@model_bp.route('/api/models/save-auto-model', methods=['POST'])
def make_auto_model():
    data = request.get_json()
    user_id = session.get('user_id')

    if not data:
        return jsonify({"error": "Invalid data format"}), 400
    
    #user already has running task
    if user_id in active_tasks:
        return jsonify({"error": "Task already running. Please wait until it finishes."}), 400

    try:
        dataset = data["dataset"]
        task = data['taskType']
        opt_method = data["optMethod"]
        print(dataset)
        print(task)
        print("Opt method: " + opt_method)
        
        active_tasks[user_id] = True
        create_notification(for_user_id = user_id, message = "Creating started")

        model = create_auto_model(dataset, task, user_id)
        time.sleep(10)

        #přesunout do funkce -> save_and_notification
        # new_model = Model(model_name = "model_"+ str(round(random.random(), 3)), accuracy = 0.75, error = 0.07, dataset = dataset, user_id = user_id)
        # db.session.add(new_model)
        # db.session.commit()
        # model_id = new_model.id

        # save_model(model, user_id, model_id)
        # create_notification(for_user_id = user_id, message = "Model creating")

        save_and_notification(model, user_id, dataset)

        #save_model(model, "userModels/" + "model.keras")
        #loaded_model = load_model("userModels/" + "model.keras")

        #loaded_model.summary()

        active_tasks.pop(user_id, None)
        return jsonify({"message": "Model successfully created and saved!"}), 200
    except Exception as e:
        active_tasks.pop(user_id, None)
        return jsonify({"error": str(e)}), 500
    

#get all models
@model_bp.route("/api/getmodels", methods=['GET'])
def return_models():
    try:
        user_id = session.get('user_id')

        # Načtení modelů pouze pro daného uživatele
        data = Model.query.filter_by(user_id=user_id).all()
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
        #model_path = f"userModels/model_{model_id}.keras"
        model_path = os.path.join("userModels", str(session.get("user_id")), f"model_{model_id}.keras")
        
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
        model = Model.query.get(model_id)

        # Ověření, zda záznam existuje
        if not model:
            return jsonify({"error": "Model not found in the database"}), 404
        
        #model_path = f"userModels/model_{model_id}.keras"  # cesta k modelu
        model_path = os.path.join("userModels", str(session.get("user_id")), f"model_{model_id}.keras")
        print(model_path)

        # Ověření, zda soubor existuje
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found"}), 404


        # Smazání záznamu z databáze
        db.session.delete(model)
        db.session.commit()

        os.remove(model_path)

        return jsonify({"message": "Model successfully deleted"}), 200
    except Exception as e:
        db.session.rollback()  # Vrácení změn v případě chyby
        return jsonify({"error": str(e)}), 500
    

#save model and create notification
def save_and_notification(model, user_id, dataset):
        try:
            new_model = Model(model_name = "model_"+ str(round(random.random(), 3)), accuracy = 0.75, error = 0.07, dataset = dataset, user_id = user_id)
            db.session.add(new_model)
            db.session.commit()
            model_id = new_model.id

            save_model(model, user_id, model_id)
            create_notification(for_user_id = user_id, message = "Model created")
        except Exception as e:
            print("Error in save_and_notification" + e)
            raise 