import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from utils.task_progress_manager import progress_manager
from .essentials import create_functional_model
from flask import session

# Funkce pro více tréninků jednoho modelu
def train_multiple_times(model, x_train, y_train, x_val, y_val, num_runs=3, threshold=0.7, monitor_metric='val_accuracy', epochs=10, batch_size=32):
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=5, min_delta=0.01, mode='max', restore_best_weights=True)
    best_epoch_history = []
    best_metric_value = 0
    best_weights = None  # Uchová váhy modelu s nejlepší finální hodnotou metriky

    for i in range(num_runs):
        try:
            epoch_history = []
            print(f"Training run {i+1}")
            history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)

            # Přidáme hodnoty metriky pro každou epochu do epoch_history
            for epoch, value in enumerate(history.history[monitor_metric]):
                epoch_history.append({"epoch": epoch + 1, "value": round(value, 3)})

            # Získáme hodnotu metriky z poslední epochy tréninku
            final_metric_value = history.history[monitor_metric][-1]

            # Pokud je finální hodnota metriky lepší než dosud nejlepší, uložíme váhy
            if final_metric_value > best_metric_value:
                best_metric_value = final_metric_value
                best_weights = model.get_weights()  # Uložení vah nejlepšího modelu
                best_epoch_history = epoch_history

            # Pokud finální metrika tréninku nedosáhla prahu threshold, ukončíme další trénování
            if final_metric_value < threshold:
                print(f"Stopping early: Model did not meet {monitor_metric} threshold of {threshold}")
                break
        except Exception as e:
            print(e)

    # Načteme nejlepší váhy zpět do modelu před jeho vrácením
    if best_weights:
        model.set_weights(best_weights)

    # Vrátíme natrénovaný model s váhami nejlepší finální metriky, finální hodnotu metriky a historii metrik
    return model, best_metric_value, best_epoch_history

# Random search pro náhodné modely
def random_search(layers, settings, x_train, y_train, x_val, y_val, num_models=5, num_runs=3, threshold=0.7, monitor_metric='val_accuracy'):
    best_model = None
    best_metric_value = 0
    best_metric_history = []
    best_model_params = []
    user_id = session.get("user_id")

    for i in range(num_models):
        print(f"Training model {i+1}")
        model, used_params = create_functional_model(layers, settings)
        
        # Trénujeme model několikrát a získáme finální hodnotu metriky a historii metrik
        trained_model, metric_value, metric_history = train_multiple_times(
            model, x_train, y_train, x_val, y_val, num_runs=num_runs, threshold=threshold, monitor_metric=monitor_metric, epochs=settings["epochs"], batch_size=settings["batch_size"]
        )
        print(f"Model {i+1} achieved {monitor_metric}: {metric_value}")
        
        # Pokud je aktuální model lepší než předchozí, uložíme ho jako nejlepší
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_model = trained_model  # Uložíme natrénovaný model s nejlepší finální metrikou
            best_metric_history = metric_history  # Uložíme historii metrik nejlepšího modelu
            best_model_params = used_params #save best params
    
        progress = ((i + 1) / num_models) * 100  # Progress jako % dokončení
        progress_manager.update_progress(user_id, progress)

    print(f"Best model achieved {monitor_metric}: {best_metric_value}")
    return best_model, best_metric_value, best_metric_history, best_model_params
