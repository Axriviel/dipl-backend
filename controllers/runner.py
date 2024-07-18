import multiprocessing
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import os

def train_model(data, model_path):
    dataset = np.loadtxt('datasets/pima-indians-diabetes.csv', delimiter=',')
    print("cesta je"+ os.getcwd() +"\n")
    # split into input (X) and output (y) variables
    X = dataset[:, 0:8]
    y = dataset[:, 8]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=8, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
            
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=25, batch_size=10, verbose=0)
    print("čarování dokončeno")
    val = model.evaluate(X_test, y_test, verbose=0)
    return val
    
    # print(f"Starting training task with data: {data}")
    # # Simulace natrénování modelu
    # model = keras.Sequential([
    #     keras.layers.Dense(10, activation='relu', input_shape=(data['input_shape'],)),
    #     keras.layers.Dense(1, activation='sigmoid')
    # ])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # # Fiktivní trénovací data
    # x_train = data['x_train']
    # y_train = data['y_train']
    
    # model.fit(x_train, y_train, epochs=10)
    # model.save(model_path)
    #time.sleep(5)
    #return("It is done")

async def run_async_task(data, model_path):
    return train_model("x", "x")
