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
