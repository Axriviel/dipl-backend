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
from utils.task_progress_manager import progress_manager
from optimizers.essentials import create_optimized_model
from config.settings import Config
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


def check_active_task(user_id):
    if user_id in active_tasks:
        return False
    else:
        active_tasks[user_id] = True
        return True

# def create_auto_model(dataset, task, opt_method, user_id ):
#     model = Sequential()
#     model.add(Dense(8))
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model



from sklearn.model_selection import train_test_split

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


        print(dataset_name)
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
        
        nni_config = settings["NNI"]
        print(nni_config)
        print("model name:", settings["model_name"])
        
        print(dataset_config)

        
        #dataset processing prováděn v create_optimized_model
        #dataset = load_dataset(dataset_path)
        #X = dataset[x_columns]
        #y = dataset[y_column]
        
        #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)


        # Vytvoření modelu
        create_notification(for_user_id=user_id, message="Creating started")
        progress_manager.update_progress(user_id, 0)
        best_model, best_metric, best_metric_history, used_params = create_optimized_model(layers, settings, dataset_path, dataset_config)

        # Uložení modelu a notifikace
        save_and_notification(best_model, user_id, dataset=dataset_name, metric_value=best_metric, watched_metric=settings["monitor_metric"], metric_values_history=best_metric_history, creation_config = [layers, settings, dataset_config], used_params = used_params,model_name = settings["model_name"], used_opt_method=settings["opt_algorithm"])
        
        active_tasks.pop(user_id, None)

        return jsonify({"message": "Model successfully created and dataset uploaded!"}), 200
    
    except Exception as e:
        active_tasks.pop(user_id, None)
        return jsonify({"error": "Error creating model " +str(e)}), 500


# #create task make model
# @model_bp.route('/api/save-model', methods=['POST'])
# def make_model():

#     #přidat kontrolu na to, aby uživatel mohl spustit pouze 1 task
#     try:
#         user_id = session.get('user_id')
#         if not user_id:
#             return jsonify({"error": "User not authenticated"}), 401

#         # Kontrola, zda soubor byl nahrán
#         if 'datasetFile' not in request.files:
#             return jsonify({"error": "No dataset file"}), 400

#         if not check_active_task(user_id):
#             return jsonify({"error": "Task already running. Please wait until it finishes."}), 400

#         file = request.files['datasetFile']
#         dataset_name = file.filename


#         dataset_path = save_dataset(file, user_id)
#         #loaded_dataset = load_dataset(dataset_path)

#         # Načtení vrstev z formuláře
#         layers = request.form.get('layers')
#         if not layers:
#             return jsonify({"error": "No model layers provided"}), 400

#         layers = json.loads(layers)  # Konverze JSON řetězce na Python dictionary
#         print(layers)

#         settings = request.form.get("settings")
#         if not settings:
#             return jsonify({"error": "No model settings provided"}), 400
        
#         settings = json.loads(settings)
#         print(settings)



#         # Vytvoření modelu
#         create_notification(for_user_id = user_id, message = "Creating started")

#         best_model, best_metric, best_metric_history = create_optimized_model(layers, settings, dataset_path)
#         print(best_model)
#         print(best_metric)
#         print(best_metric_history)
        
#         # model.summary()

#         # Uložení modelu a notifikace
#         #změnit na dataset_name
#         save_and_notification(best_model, user_id, dataset = dataset_name, metric_value = best_metric, watched_metric = settings["monitor_metric"], metric_values_history = best_metric_history)
        
#         active_tasks.pop(user_id, None)

#         return jsonify({"message": "Model successfully created and dataset uploaded!"}), 200
    
#     except Exception as e:
#         active_tasks.pop(user_id, None)
#         return jsonify({"error": str(e)}), 500
    
#create task make model


#create model with full auto method
#přidal jsem timeout na frontend - zahrnout limity
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

        dataset_name = request.form.get("datasetFile")
        print(dataset_name)
        if not dataset_name:
            return jsonify({"error": "No dataset name provided"}), 400
        # file = request.files['datasetFile']

        active_tasks[user_id] = True

        # dataset_path = save_dataset(file, user_id)


        # Přečteme taskType a optMethod z formulářových dat
        layers = json.loads(request.form.get('layers', '[]'))
        settings = json.loads(request.form.get('settings', '{}'))
        dataset_config = json.loads(request.form.get('datasetConfig', '{}'))
        max_models = request.form.get('maxModels')
        timer = request.form.get('timeOut')
        task_type = request.form.get('taskType')
        use_default_dataset = request.form.get('useDefaultDataset')
        tags = request.form.get("tags")
        print("tasg:", tags)
        print("default dataset:", use_default_dataset)

    
        print(f"Task Type: {task_type}")
        print(f"Optimization Method: {settings["opt_algorithm"]}")
        print("layers auto", layers)
        print("settings auto", settings)
        print("dataset_config auto", dataset_config)
        print("max_models auto", max_models)
        print("timer auto", timer)

        additional_data = {"task_type": task_type, "tags": tags }
        print(additional_data, "jsou další data")
        #read dataset


        if use_default_dataset == "true":
            dataset_path = os.path.join(Config.DEFAULT_DATASET_FOLDER, dataset_name)
        else:
            dataset_path = os.path.join(Config.DATASET_FOLDER, str(user_id), dataset_name)

        if not os.path.exists(dataset_path):
            return jsonify({"error": "Dataset not found"}), 400

        # dataset = load_dataset(dataset_path)


        
        
        create_notification(for_user_id = user_id, message = "Creating started")
        #model = create_auto_model(dataset, task_type, opt_method, user_id)
        progress_manager.update_progress(user_id, 0)
        best_model, best_metric, best_metric_history, used_params = create_optimized_model(layers, settings, dataset_path, dataset_config, opt_data=additional_data)

        # Uložení modelu a notifikace
        save_and_notification(best_model, user_id, dataset=dataset_name, metric_value=best_metric, watched_metric=settings["monitor_metric"], metric_values_history=best_metric_history, creation_config = [layers, settings, dataset_config], used_params = used_params, model_name = settings["model_name"], used_opt_method=settings["opt_algorithm"], used_task = task_type, used_tags = tags)
        

        
        # tohle bude možné použít pravděpodobně
        #best_model, best_metric, best_metric_history, used_params = create_optimized_model(layers, settings, dataset_path, dataset_config)

        #přesunout do funkce -> save_and_notification
        # new_model = Model(model_name = "model_"+ str(round(random.random(), 3)), accuracy = 0.75, error = 0.07, dataset = dataset, user_id = user_id)
        # db.session.add(new_model)
        # db.session.commit()
        # model_id = new_model.id
        # save_model(model, user_id, model_id)
        # create_notification(for_user_id = user_id, message = "Model creating")
        #save_and_notification(model, user_id, dataset_name)
        #save_model(model, "userModels/" + "model.keras")
        #loaded_model = load_model("userModels/" + "model.keras")
        #loaded_model.summary()
        active_tasks.pop(user_id, None)
        #upravit, abych vrátil info "model se vytváří" a uložení a notifikaci udělat mimo
        return jsonify({"message": "Model successfully created and saved!"}), 200
    except Exception as e:
        active_tasks.pop(user_id, None)
        return jsonify({"error": "Error creating model " + str(e)}), 500
    

#get all models
@model_bp.route("/api/getmodels", methods=['GET'])
def return_models():
    try:
        user_id = session.get('user_id')

        # Načtení modelů pouze pro daného uživatele
        data = Model.query.filter_by(user_id=user_id).all()
                # Serializace dat do seznamu slovníků

        from DTO.mapper import map_model_to_dto
        model_list = [
            map_model_to_dto(model).to_dict()
            for model in data
        ]
        
        # Vrácení dat jako JSON odpověď
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
            layer_info = {
                'layer_name': layer.name,
                'layer_type': layer.__class__.__name__,
                #'output_shape': layer.shape,  # Výstupní tvar vrstvy
                'num_params': layer.count_params(),  # Počet parametrů
                'trainable': layer.trainable  # Jestli je vrstva trénovatelná
            }
            layers_info.append(layer_info)

        # Souhrn modelu v JSON formátu
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
        # Načtení modelu z databáze podle ID
        model = Model.query.get(model_id)
        
        if not model:
            return jsonify({"error": "Model not found"}), 404
        
        # Vrácení požadovaných dat jako JSON odpověď
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
            db.session.delete(model)
            db.session.commit()
            return jsonify({"error": "Model not found"}), 409


        # Smazání záznamu z databáze
        db.session.delete(model)
        db.session.commit()

        os.remove(model_path)

        return jsonify({"message": "Model successfully deleted"}), 200
    except Exception as e:
        db.session.rollback()  # Vrácení změn v případě chyby
        return jsonify({"error": str(e)}), 500
    
@model_bp.route('/api/task-progress', methods=['GET'])
def get_task_progress():
    """ Vrátí stav progress baru a stav úlohy pro přihlášeného uživatele """
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not authenticated"}), 401

    progress = progress_manager.get_progress(user_id)
    
    # Zkontrolujeme, zda úloha stále běží
    is_running = user_id in active_tasks  # Pokud je ve `active_tasks`, znamená to, že běží

    return jsonify({
        "progress": progress if progress is not None else 0,
        "isRunning": is_running
    })


#save model and create notification
def save_and_notification(model, user_id, dataset, metric_value="0", watched_metric="accuracy", metric_values_history=[{}], creation_config = [{}], used_params=[{}], model_name = "myModel", used_opt_method="undefined", used_task = "", used_tags = {}):
        try:
            if model_name == "myModel":
                model_name = "model_"+ str(round(random.random(), 3))

            new_model = Model(model_name = model_name, accuracy = 0.75, metric_value = round(metric_value, 3), watched_metric = watched_metric, metric_values_history = metric_values_history, creation_config = creation_config, used_params = used_params, used_opt_method=used_opt_method, error = 0.07, dataset = dataset, user_id = user_id, used_task=used_task, used_tags=used_tags)
            db.session.add(new_model)
            db.session.commit()
            model_id = new_model.id

            save_model(model, user_id, model_id)
            create_notification(for_user_id = user_id, message = "Model "+ new_model.model_name +" created")
        except Exception as e:
            print("Error in save_and_notification" + str(e))
            raise 