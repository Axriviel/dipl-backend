import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from .essentials import create_functional_model

# Funkce pro více tréninků jednoho modelu
def train_multiple_times(model, x_train, y_train, x_val, y_val, num_runs=3, threshold=0.7, monitor_metric='val_accuracy'):
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=5, min_delta=0.01, mode='max', restore_best_weights=True)
    metric_values = []
    
    for i in range(num_runs):
        try:
            print(f"Training run {i+1}")
            history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

            # Zjistíme nejlepší dosaženou hodnotu sledované metriky
            max_metric_value = max(history.history[monitor_metric])
            metric_values.append(max_metric_value)

            # Pokud metrika přesahuje stanovený threshold, pokračujeme
            if max_metric_value < threshold:
                print(f"Stopping early: Model did not meet {monitor_metric} threshold of {threshold}")
                break
        except Exception as e:
            print(e)
    return sum(metric_values) / len(metric_values) if metric_values else 0

# Random search pro náhodné modely
# def random_search(layers, input_shape, x_train, y_train, x_val, y_val, num_models=5, num_runs=3, threshold=0.7, metrics=['accuracy'], monitor_metric='val_accuracy'):
def random_search(layers, settings, x_train, y_train, x_val, y_val, num_models=5, num_runs=3, threshold=0.7, monitor_metric='val_accuracy'):
    best_model = None
    best_metric_value = 0

    for i in range(num_models):
        print(f"Training model {i+1}")
        model = create_functional_model(layers, settings)
        
        # Trénujeme model několikrát a získáme průměrnou hodnotu metriky
        metric_value = train_multiple_times(model, x_train, y_train, x_val, y_val, num_runs=num_runs, threshold=threshold, monitor_metric=monitor_metric)
        print(f"Model {i+1} achieved {monitor_metric}: {metric_value}")

        # Pokud je aktuální model lepší než předchozí, uložíme ho jako nejlepší
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_model = model

    print(f"Best model achieved {monitor_metric}: {best_metric_value}")
    return best_model, best_metric_value
