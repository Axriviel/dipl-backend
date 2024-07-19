import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras

from flask import Flask, jsonify, request
import asyncio
import uuid

import controllers.runner
from repositories.UserRepository import UserRepository
from services.UserService import UserService

app = Flask(__name__)

# Inicializace UserRepository a UserService
user_repository = UserRepository()
user_service = UserService(user_repository)



user_results = {}

@app.route('/test', methods=['GET'])
def get_users():

    users = user_service.getAll()
    #users_list = [{'id': user.id, 'username': user.username, 'email': user.email} for user in users]
    return jsonify(users)


# Endpoint pro zpracování požadavků
@app.route('/', methods=['GET'])
async def calculate():
    # data = request.get_json()
    # params = data['params']  # předpokládáme, že získáme parametry z requestu
    # user_id = str(uuid.uuid4())
    user_id = str(5)
    results = []

    task = asyncio.create_task(controllers.runner.run_async_task("xx", "xd"))
    user_results[user_id] = task

    return jsonify({'user_id': user_id}), 202



@app.route('/result/<user_id>', methods=['GET'])
async def get_result(user_id):
    # return(user_results)
    if user_id in user_results:
        task = user_results[user_id]

        if task.done():
            result = task.result()
            del user_results[user_id]  # odstraníme výsledek po vrácení
            print(result)
            # return result
            return jsonify({'result': result})
        else:
            return jsonify({'status': 'running'}), 202  # vrátíme status, že výpočet stále běží
    else:
        return jsonify({'error': 'Výsledek pro zadané ID nenalezen'}), 404

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)