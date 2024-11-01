# blueprints/model_routes.py
from flask import Blueprint, jsonify, request
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Dense
from utils.model_storing import save_model, load_model
from flask import send_file
from controllers.notification_controller import create_notification
from controllers.user_controller import session
from models.model import Model, db
from utils.dataset_storing import save_dataset, load_dataset
from optimizers.essentials import create_optimized_model
import random
import os
import time
import json


model_bp = Blueprint('model_bp', __name__)


active_tasks = {}

# def create_model(layers, dataset):
#     model = Sequential()
#     for layer in layers:
#         if layer['type'] == 'Dense':
#             model.add(Dense(
#                 units=layer['units'],
#                 activation=layer['activation'],
#                 input_shape=layer.get('input_shape', None)
#             ))

#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model


def create_auto_model(dataset, task, opt_method, user_id ):
    model = Sequential()
    model.add(Dense(8))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



#create task make model
@model_bp.route('/api/save-model', methods=['POST'])
def make_model():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        # Kontrola, zda soubor byl nahrán
        if 'datasetFile' not in request.files:
            return jsonify({"error": "No dataset file"}), 400

        file = request.files['datasetFile']
        # dataset_name = file.filename


        dataset_path = save_dataset(file, user_id)
        #loaded_dataset = load_dataset(dataset_path)

        # Načtení vrstev z formuláře
        layers = request.form.get('layers')
        if not layers:
            return jsonify({"error": "No model layers provided"}), 400

        layers = json.loads(layers)  # Konverze JSON řetězce na Python dictionary
        print(layers)

        settings = request.form.get("settings")
        if not settings:
            return jsonify({"error": "No model settings provided"}), 400
        
        settings = json.loads(settings)
        print(settings)



        # Vytvoření modelu
        create_notification(for_user_id = user_id, message = "Creating started")

        best_model, best_metric, best_metric_history = create_optimized_model(layers, settings, dataset_path)
        print(best_model)
        print(best_metric)
        print(best_metric_history)
        
        # model.summary()

        # Uložení modelu a notifikace
        #změnit na dataset_name
        save_and_notification(best_model, user_id, dataset = dataset_path, metric_value = best_metric, watched_metric = settings["monitor_metric"], metric_values_history = best_metric_history)

        return jsonify({"message": "Model successfully created and dataset uploaded!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
#create task make model
@model_bp.route('/api/models/save-auto-model', methods=['POST'])
def make_auto_model():
    #data = request.get_json()
    try:
        user_id = session.get('user_id')

        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401
        
        #user already has running task
        if user_id in active_tasks:
            return jsonify({"error": "Task already running. Please wait until it finishes."}), 400

        if 'datasetFile' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['datasetFile']
        dataset_name = file.filename

        active_tasks[user_id] = True

        dataset_path = save_dataset(file, user_id)
        # Přečteme taskType a optMethod z formulářových dat
        task_type = request.form.get('taskType')
        opt_method = request.form.get('optMethod')
        print(f"Task Type: {task_type}")
        print(f"Optimization Method: {opt_method}")

        #read dataset
        dataset = load_dataset(dataset_path)
        
        create_notification(for_user_id = user_id, message = "Creating started")
        model = create_auto_model(dataset, task_type, opt_method, user_id)
        time.sleep(10)
        #přesunout do funkce -> save_and_notification
        # new_model = Model(model_name = "model_"+ str(round(random.random(), 3)), accuracy = 0.75, error = 0.07, dataset = dataset, user_id = user_id)
        # db.session.add(new_model)
        # db.session.commit()
        # model_id = new_model.id
        # save_model(model, user_id, model_id)
        # create_notification(for_user_id = user_id, message = "Model creating")
        save_and_notification(model, user_id, dataset_name)
        #save_model(model, "userModels/" + "model.keras")
        #loaded_model = load_model("userModels/" + "model.keras")
        #loaded_model.summary()
        active_tasks.pop(user_id, None)
        #upravit, abych vrátil info "model se vytváří" a uložení a notifikaci udělat mimo
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

                #v podstatě viewmodel - definovat mimo
        model_list = [
            {
                'id': model.id,
                'name': model.model_name,
                "accuracy": model.accuracy,
                "metric_value": model.metric_value,
                "watched_metric": model.watched_metric,
                "metric_values_history": model.metric_values_history,
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
def save_and_notification(model, user_id, dataset, metric_value="0", watched_metric="accuracy", metric_values_history=[{}]):
        try:
            new_model = Model(model_name = "model_"+ str(round(random.random(), 3)), accuracy = 0.75, metric_value = metric_value, watched_metric = round(watched_metric, 3), metric_values_history = metric_values_history, error = 0.07, dataset = dataset, user_id = user_id)
            db.session.add(new_model)
            db.session.commit()
            model_id = new_model.id

            save_model(model, user_id, model_id)
            create_notification(for_user_id = user_id, message = "Model "+ new_model.model_name +" created")
        except Exception as e:
            print("Error in save_and_notification" + str(e))
            raise 