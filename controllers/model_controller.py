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
from utils.task_progress_manager import progress_manager, termination_manager
from utils.time_limit_manager import time_limit_manager
from utils.task_protocol_manager import task_protocol_manager
from optimizers.essentials import create_optimized_model
from config.settings import Config
from dataclasses import asdict
import random
import os
import time
from datetime import datetime
import json


model_bp = Blueprint('model_bp', __name__)


active_tasks = {}

def check_active_task(user_id):
    if user_id in active_tasks:
        return False
    else:
        active_tasks[user_id] = True
        return True

def reset_managers(user_id):
    active_tasks.pop(user_id, None)
    progress_manager.reset_user(user_id)
    termination_manager.reset_user(user_id)
    time_limit_manager.reset_user(user_id)
    task_protocol_manager.reset_user(user_id)

# used for semi automatic model creation
@model_bp.route('/api/save-model', methods=['POST'])
def make_model():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        if not check_active_task(user_id):
            return jsonify({"error": "Task already running. Please wait until it finishes."}), 400


        dataset_name = request.form.get("datasetFile")
        use_default_dataset = request.form.get('useDefaultDataset')

        if not dataset_name:
            return jsonify({"error": "No dataset name provided"}), 400
        
        if use_default_dataset == "true":
            dataset_path = os.path.join(Config.DEFAULT_DATASET_FOLDER, dataset_name)
        else:
            dataset_path = os.path.join(Config.DATASET_FOLDER, str(user_id), dataset_name)

        if not os.path.exists(dataset_path):
            return jsonify({"error": "Dataset not found"}), 400

        # load layers, settings and dataset_settings from form
        layers = json.loads(request.form.get('layers', '[]'))
        settings = json.loads(request.form.get('settings', '{}'))
        dataset_config = json.loads(request.form.get('datasetConfig', '{}'))
        tags = request.form.get("tags")

        # task type is known for custom designer
        additional_data = {"task_type": "undefined", "tags": tags }
        
        if(settings["use_timeout"]):
            time_limit_manager.add_user(user_id, settings["timeout"])
        
        # create model
        create_notification(for_user_id=user_id, message="Creating started")
        progress_manager.update_progress(user_id, 0)
        task_protocol_manager.log_item(user_id, "started_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        best_model, best_metric, best_metric_history, used_params = create_optimized_model(layers, settings, dataset_path, dataset_config,  opt_data=additional_data)

        # save model and create notification
        save_and_notification(best_model, user_id, dataset=dataset_name, metric_value=best_metric, watched_metric=settings["monitor_metric"], metric_values_history=best_metric_history, creation_config = [layers, settings, dataset_config], used_params = used_params,model_name = settings["model_name"], used_opt_method=settings["opt_algorithm"], used_task = "undefined", used_tags = tags, used_designer="custom")
        
        reset_managers(user_id)
        return jsonify({"message": "Model successfully created and dataset uploaded!"}), 200
    
    except IndexError as e:
        reset_managers(user_id)
        print(e)
        create_notification(for_user_id=user_id, message=""+str(e))
        return jsonify({"error": "Incorrect dataset config column indexes"}), 500 

    except Exception as e:
        reset_managers(user_id)
        print(e)
        create_notification(for_user_id=user_id, message=""+str(e))
        return jsonify({"error": "Error creating model: " +str(e)}), 500

#create model with full auto method
@model_bp.route('/api/models/save-auto-model', methods=['POST'])
def make_auto_model():
    try:
        user_id = session.get('user_id')

        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401
        
        #user already has running task
        if user_id in active_tasks:
            return jsonify({"error": "Task already running. Please wait until it finishes."}), 400

        dataset_name = request.form.get("datasetFile")
        print(dataset_name)
        if not dataset_name:
            return jsonify({"error": "No dataset name provided"}), 400

        active_tasks[user_id] = True

        layers = json.loads(request.form.get('layers', '[]'))
        settings = json.loads(request.form.get('settings', '{}'))
        dataset_config = json.loads(request.form.get('datasetConfig', '{}'))
        max_models = request.form.get('maxModels')
        task_type = request.form.get('taskType')
        use_default_dataset = request.form.get('useDefaultDataset')
        tags = request.form.get("tags")
        # print("tasg:", tags)
        # print("default dataset:", use_default_dataset)

    
        # print(f"Task Type: {task_type}")
        # print(f"Optimization Method: {settings["opt_algorithm"]}")
        # print("layers auto", layers)
        # print("settings auto", settings)
        # print("dataset_config auto", dataset_config)
        # print("max_models auto", max_models)

        additional_data = {"task_type": task_type, "tags": tags }
        # print(additional_data, "jsou další data")
        #read dataset


        if use_default_dataset == "true":
            dataset_path = os.path.join(Config.DEFAULT_DATASET_FOLDER, dataset_name)
        else:
            dataset_path = os.path.join(Config.DATASET_FOLDER, str(user_id), dataset_name)

        if not os.path.exists(dataset_path):
            return jsonify({"error": "Dataset not found"}), 400

        if(settings["use_timeout"]):
            print("using timer", settings["use_timeout"])
            time_limit_manager.add_user(user_id, settings["timeout"])
        create_notification(for_user_id = user_id, message = "Creating started")

        task_protocol_manager.log_item(user_id, "started_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        progress_manager.update_progress(user_id, 0)
        best_model, best_metric, best_metric_history, used_params = create_optimized_model(layers, settings, dataset_path, dataset_config, opt_data=additional_data)

        save_and_notification(best_model, user_id, dataset=dataset_name, metric_value=best_metric, watched_metric=settings["monitor_metric"], metric_values_history=best_metric_history, creation_config = [layers, settings, dataset_config], used_params = used_params, model_name = settings["model_name"], used_opt_method=settings["opt_algorithm"], used_task = task_type, used_tags = tags, used_designer="automated")
        

        reset_managers(user_id)

        return jsonify({"message": "Model successfully created and saved!"}), 200
    except IndexError as e:
        reset_managers(user_id)
        create_notification(for_user_id=user_id, message=""+str(e))
        return jsonify({"error": "Incorrect dataset config values"}), 500 
    except Exception as e:
        reset_managers(user_id)
        create_notification(for_user_id=user_id, message=""+str(e))
        return jsonify({"error": "Error creating model: " + str(e)}), 500
    

#get all models
@model_bp.route("/api/getmodels", methods=['GET'])
def return_models():
    try:
        user_id = session.get('user_id')

        data = Model.query.filter_by(user_id=user_id).all()

        from DTO.mapper import map_model_to_dto
        model_list = [
            map_model_to_dto(model).to_dict()
            for model in data
        ]
        
        return jsonify(model_list)
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500
    

#get model details
@model_bp.route('/api/get-details/<int:model_id>', methods=['GET'])
def get_model_details(model_id):
    try:
        model_path = os.path.join("userModels", str(session.get("user_id")), f"model_{model_id}.keras")
        loaded_model = load_model(model_path)

        layers_info = []
        for layer in loaded_model.layers:
            neurons = None
            if hasattr(layer, 'units'):      
                neurons = layer.units
            elif hasattr(layer, 'filters'): 
                neurons = layer.filters
            layer_info = {
                'layer_name': layer.name,
                'layer_type': layer.__class__.__name__,
                #'output_shape': layer.shape,  # Výstupní tvar vrstvy
                'num_params': layer.count_params(),  # Počet parametrů
                'neurons': neurons,
                'trainable': layer.trainable 
            }
            layers_info.append(layer_info)

        model_summary = {
            'model_name': loaded_model.name,
            'layers': layers_info
        }
        return jsonify(model_summary), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
#test param detail
@model_bp.route('/api/get-params/<int:model_id>', methods=['GET'])
def get_param_details(model_id):
    try:
        model = Model.query.get(model_id)
        
        if not model:
            return jsonify({"error": "Model not found"}), 404
        
        return jsonify({
            "creation_config": model.creation_config,
            "used_params": model.used_params
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#download selected model
@model_bp.route('/api/download-model/<int:model_id>', methods=['GET'])
def download_model(model_id):
    try:
        model_path = os.path.join("userModels", str(session.get("user_id")), f"model_{model_id}.keras")
        
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

        if not model:
            return jsonify({"error": "Model not found in the database"}), 404
        
        model_path = os.path.join("userModels", str(session.get("user_id")), f"model_{model_id}.keras")

        if not os.path.exists(model_path):
            db.session.delete(model)
            db.session.commit()
            return jsonify({"error": "Model not found"}), 409


        db.session.delete(model)
        db.session.commit()

        os.remove(model_path)

        return jsonify({"message": "Model successfully deleted"}), 200
    except Exception as e:
        db.session.rollback()  # rollback in case of error
        return jsonify({"error": str(e)}), 500
    
@model_bp.route('/api/task-progress', methods=['GET'])
def get_task_progress():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not authenticated"}), 401

    progress = progress_manager.get_progress(user_id)
    
    is_running = user_id in active_tasks  

    return jsonify({
        "progress": progress if progress is not None else 0,
        "isRunning": is_running
    })

@model_bp.route("/api/cancel-task", methods=['GET'])
def cancel_user_task():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not authenticated"}), 401
    termination_manager.terminate_user_task(user_id)
    return jsonify({"message": "Task cancel accepted"}), 200


#save model and create notification
def save_and_notification(model, user_id, dataset, metric_value="0", watched_metric="accuracy", metric_values_history=[{}], creation_config = [{}], used_params=[{}], model_name = "", used_opt_method="undefined", used_task = "", used_tags = {}, used_designer="unknown"):
        try:
            task_protocol_manager.log_item(user_id, "finished_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            if model_name == "":
                model_name = "model_"+ str(round(random.random(), 3))

            new_model = Model(model_name = model_name, metric_value = round(metric_value, 3), watched_metric = watched_metric, metric_values_history = metric_values_history, creation_config = creation_config, used_params = used_params, used_opt_method=used_opt_method, dataset = dataset, user_id = user_id, used_task=used_task, used_tags=used_tags, used_designer=used_designer, task_protocol=asdict(task_protocol_manager.get_log(user_id)))
            db.session.add(new_model)
            db.session.commit()
            model_id = new_model.id

            save_model(model, user_id, model_id)
            create_notification(for_user_id = user_id, message = "Model "+ new_model.model_name +" created")
        except Exception as e:
            print("Error in save_and_notification" + str(e))
            raise 