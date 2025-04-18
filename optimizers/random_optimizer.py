import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from utils.task_progress_manager import progress_manager, termination_manager
from utils.time_limit_manager import time_limit_manager
from .essentials import create_functional_model
from flask import session
from utils.task_progress_manager import growth_limiter_manager
from utils.task_progress_manager import ExternalTerminationCallback
from utils.task_protocol_manager import task_protocol_manager
import warnings

def train_multiple_times(model, x_train, y_train, x_val, y_val, num_runs=3, threshold=0.7, monitor_metric='val_accuracy', epochs=10, batch_size=32, user_id = "", es_patience=10, es_delta=0.01):
    try:

        warnings.filterwarnings("error", category=UserWarning)
        early_stopping = EarlyStopping(monitor=monitor_metric, patience=es_patience, min_delta=es_delta, mode='max', restore_best_weights=True)
        external_termination = ExternalTerminationCallback(user_id=user_id)

        best_epoch_history = []
        best_metric_value = 0
        best_weights = None

        for i in range(num_runs):
            try:
                epoch_history = []
                history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, external_termination], verbose=1)

                for epoch, value in enumerate(history.history[monitor_metric]):
                    epoch_history.append({"epoch": epoch + 1, "value": round(value, 3)})

                final_metric_value = history.history[monitor_metric][-1]

                if final_metric_value > best_metric_value:
                    best_metric_value = final_metric_value
                    best_weights = model.get_weights()
                    best_epoch_history = epoch_history

                if final_metric_value < threshold:
                    print(f"Stopping early: Model did not meet {monitor_metric} threshold of {threshold}")
                    break

                if(termination_manager.is_terminated(user_id)):
                    task_protocol_manager.log_item(user_id, "stopped_by", "user")
                    raise Exception("Task terminated by user")
                if(time_limit_manager.has_time_exceeded(user_id)):
                    task_protocol_manager.log_item(user_id, "stopped_by", "timeout")
                    print("ending on time")
                    break

            except Exception as e:
                raise e

        if best_weights:
            model.set_weights(best_weights)

        return model, best_metric_value, best_epoch_history
    except Exception as e:
        raise e
    
# Random search
def random_search(layers, settings, x_train, y_train, x_val, y_val, num_models=5, num_runs=3, threshold=0.7, monitor_metric='val_accuracy', trackProgress = True):
    try:
        print("gonna create", num_models)
        best_model = None
        best_metric_value = 0
        best_metric_history = []
        best_model_params = []
        user_id = session.get("user_id")

        for i in range(num_models):
            # in random optimizer one model is considered one epoch
            # create record of the epoch
            task_protocol_manager.get_log(user_id).get_or_create_epoch(epoch_number = (i+1))
            model, used_params = create_functional_model(layers, settings)

            trained_model, metric_value, metric_history = train_multiple_times(
                model, x_train, y_train, x_val, y_val, num_runs=num_runs, threshold=threshold, monitor_metric=monitor_metric, epochs=settings["epochs"], batch_size=settings["batch_size"], user_id = user_id, es_patience=settings.get("es_patience", 10), es_delta=settings.get("es_delta", 0.01)
            )

            layers_info = []
            for layer in model.layers:
                neurons = None
                if hasattr(layer, 'units'):       # Dense, LSTM, atd.
                    neurons = layer.units
                elif hasattr(layer, 'filters'):   # Conv vrstvy
                    neurons = layer.filters
                layer_info = {
                    'layer_name': layer.name,
                    'layer_type': layer.__class__.__name__,
                    'num_params': layer.count_params(),
                    'trainable': layer.trainable,
                    'neurons': neurons
                }
                try:
                    layer_info['output_shape'] = str(layer.output_shape)
                except AttributeError:
                    layer_info['output_shape'] = "N/A"
                layers_info.append(layer_info)
            
            task_protocol_manager.get_log(user_id).log_model_to_epoch(
            epoch_number = i+1,
            model_id = i+1,
            architecture = layers_info,
            history=metric_history,
            parameters = used_params,
            results = metric_value
            )

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_model = trained_model  
                best_metric_history = metric_history  
                best_model_params = used_params 
            if(trackProgress):
                progress = ((i + 1) / num_models) * 100  # Progress jako % dokončení
                progress_manager.update_progress(user_id, progress)
            if(termination_manager.is_terminated(user_id)):
                task_protocol_manager.log_item(user_id, "stopped_by", "user")
                raise Exception("Task terminated by user")
            if(time_limit_manager.has_time_exceeded(user_id)):
                task_protocol_manager.log_item(user_id, "stopped_by", "timeout")
                print("ending on time")
                break
            # increase the progress for growth_limiter
            growth_limiter_manager.update_progress(user_id)
            
        return best_model, best_metric_value, best_metric_history, best_model_params
    except Exception as e:
        raise e