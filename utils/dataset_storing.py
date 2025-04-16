from config.settings import Config
from flask import jsonify
import os
import pandas as pd
import numpy as np

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def save_dataset(file, user_id):
    if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or missing file"}), 400

        # Uložení datasetu do složky podle user_id
    user_folder = os.path.join(Config.DATASET_FOLDER, str(user_id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)    
    dataset_path = os.path.join(user_folder, file.filename)
    file.save(dataset_path)

    return dataset_path

def load_dataset(path):
    try:
        if path.endswith('.csv'):
            dataset = pd.read_csv(path)
        elif path.endswith('.tsv'):
            dataset = pd.read_csv(path, delimiter='\t')
        elif path.endswith('.npy'):
            dataset = np.load(path)
        elif path.endswith('.npz'):
            dataset = np.load(path)
        elif path.endswith('.h5'):
            dataset = pd.read_hdf(path)
        return dataset
    except Exception as e:
        print("Error in loading dataset", e)
        raise
