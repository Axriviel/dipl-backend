# blueprints/model_routes.py
from flask import Blueprint, jsonify, request

from flask import send_file
from controllers.notification_controller import create_notification
from controllers.user_controller import session
from utils.dataset_storing import save_dataset, load_dataset
from optimizers.essentials import create_optimized_model
import os
from config.settings import Config

dataset_bp = Blueprint('dataset_bp', __name__)


#accept and save dataset
@dataset_bp.route('/api/dataset/save-dataset', methods=['POST'])
def save_dataset_endpoint():
    """
    Endpoint pro nahrání datasetu a jeho uložení.
    """
    try:
        # Kontrola, zda je uživatel přihlášen
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        # Kontrola, zda soubor byl nahrán
        if 'datasetFile' not in request.files:
            return jsonify({"error": "No dataset file"}), 400

        file = request.files['datasetFile']
        dataset_name = file.filename

        # Uložení datasetu
        dataset_path = save_dataset(file, user_id)

        return jsonify({"message": "Dataset successfully uploaded!", "dataset_name": dataset_name, "dataset_path": dataset_path}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

#return users datasets
@dataset_bp.route('/api/dataset/list-datasets', methods=['GET'])
def list_datasets():
    """
    Endpoint pro získání seznamu všech datasetů uložených pro přihlášeného uživatele.
    """
    try:
        # Ověření přihlášení uživatele
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        # Cesta ke složce s datasetem uživatele
        user_folder = os.path.join(Config.DATASET_FOLDER, str(user_id))

        # Ověření, zda složka existuje
        if not os.path.exists(user_folder):
            return jsonify({"datasets": []})  # Pokud složka neexistuje, vrátíme prázdný seznam

        # Načtení seznamu souborů
        dataset_files = os.listdir(user_folder)

        # Vrácení seznamu datasetů
        return jsonify({"datasets": dataset_files}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dataset_bp.route('/api/dataset/delete', methods=['DELETE'])
def delete_dataset():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        data = request.get_json()
        dataset_name = data.get("dataset_name")

        if not dataset_name:
            return jsonify({"error": "Dataset name is required"}), 400

        dataset_path = os.path.join(Config.DATASET_FOLDER, str(user_id), dataset_name)

        if not os.path.exists(dataset_path):
            return jsonify({"error": "Dataset not found"}), 404

        os.remove(dataset_path)
        return jsonify({"message": f"Dataset '{dataset_name}' was deleted successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dataset_bp.route('/api/dataset/upload', methods=['POST'])
def upload_dataset():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        if 'datasetFile' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['datasetFile']
        dataset_path = save_dataset(file, user_id)

        return jsonify({"message": f"Dataset '{file.filename}' uploaded successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@dataset_bp.route('/api/dataset/details', methods=['POST'])
def get_dataset_details():
    """Vrátí podrobnosti o datasetu"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401

        data = request.get_json()
        dataset_name = data.get("dataset_name")

        if not dataset_name:
            return jsonify({"error": "Dataset name is required"}), 400

        dataset_path = get_dataset_path(user_id, dataset_name)
        if not dataset_path:
            return jsonify({"error": "Dataset not found"}), 404

        dataset_info = getDatasetInfo(dataset_path, dataset_name)

        if "error" in dataset_info:
            return jsonify(dataset_info), 400

        return jsonify(dataset_info), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@dataset_bp.route('/api/dataset/get_column_names', methods=['POST'])
def get_column_names():
    """Vrátí seznam názvů sloupců datasetu"""
    data = request.get_json()
    dataset_name = data.get("dataset_name")    
    is_default_dataset = data.get("is_default_dataset")    
    user_id = session.get('user_id')

    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    if not user_id:
        return jsonify({"error": "User not authenticated"}), 401
    print(is_default_dataset)
    dataset_path = get_dataset_path(user_id, dataset_name, is_default_dataset)
    print(dataset_path)
    if not dataset_path:
        return jsonify({"error": "Dataset not found"}), 404

    dataset_info = getDatasetInfo(dataset_path, dataset_name)
    # print(dataset_info["column_names"])

    if "error" in dataset_info:
        return jsonify(dataset_info), 400

    return jsonify({"columns": dataset_info["column_names"]}), 200


def get_dataset_path(user_id, dataset_name, is_default_dataset = False):
    if(is_default_dataset):
        return os.path.join(Config.DATASET_FOLDER, "default", dataset_name)

    """Vrátí cestu k datasetu uživatele"""
    user_folder = os.path.join(Config.DATASET_FOLDER, str(user_id))
    dataset_path = os.path.join(user_folder, dataset_name)
    return dataset_path if os.path.exists(dataset_path) else None

def getDatasetInfo(dataset_path, dataset_name):
    """Načte dataset a vrátí informace o něm"""
    dataset_info = {
        "dataset_name": dataset_name,
        "num_rows": None,
        "num_columns": None,
        "shape": None,
        "column_names": None,
    }

    try:
        import pandas as pd
        import numpy as np
        if dataset_path.endswith('.csv') or dataset_path.endswith('.tsv'):
            sep = ',' if dataset_path.endswith('.csv') else '\t'
            loaded_dataset = pd.read_csv(dataset_path, sep=sep)

            dataset_info["num_rows"] = {"rows":loaded_dataset.shape[0]}
            dataset_info["num_columns"] = {"columns": loaded_dataset.shape[1]}
            dataset_info["shape"] = {"shape":loaded_dataset.shape}
            dataset_info["column_names"] = list(loaded_dataset.columns)

        elif dataset_path.endswith('.npy'):
            pass
            # np_data = np.load(dataset_path)
            # if np_data.ndim == 1:
            #     np_data = np_data.reshape(-1, 1)  # Pokud je jednorozměrné, udělej z něj sloupec
            # dataset = pd.DataFrame(np_data, columns=[f"Column_{i}" for i in range(np_data.shape[1])])

            # dataset_info["num_rows"] = dataset.shape[0]
            # dataset_info["num_columns"] = dataset.shape[1]
            # dataset_info["shape"] = dataset.shape
            # dataset_info["column_names"] = list(dataset.columns)
            # dataset_info["sample_data"] = dataset.head(5).to_dict(orient="records")

        elif dataset_path.endswith('.npz'):
            loaded_dataset = np.load(dataset_path)
            dataset_info["num_rows"] = {k: v.shape[0] for k, v in loaded_dataset.items()}
            dataset_info["num_columns"] = {k: 1 for k, v in loaded_dataset.items()}  # Každé 1D pole = 1 sloupec
            dataset_info["shape"] = {k: v.shape for k, v in loaded_dataset.items()}
            dataset_info["column_names"] = list(loaded_dataset.keys())

        elif dataset_path.endswith('.h5'):
            pass
            # with h5py.File(dataset_path, 'r') as f:
            #     dataset_info["column_names"] = list(f.keys())
            #     dataset_info["num_rows"] = {k: f[k].shape[0] for k in f.keys()}
            #     dataset_info["num_columns"] = {k: f[k].shape[1] if len(f[k].shape) > 1 else 1 for k in f.keys()}
            #     dataset_info["shape"] = {k: f[k].shape for k in f.keys()}

        else:
            return {"error": "Unsupported dataset format"}
        
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}

    return dataset_info