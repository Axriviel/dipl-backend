from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from flask import session
from utils.task_protocol_manager import task_protocol_manager
from utils.task_progress_manager import progress_manager, termination_manager, growth_limiter_manager
from utils.time_limit_manager import time_limit_manager
from .random_optimizer import train_multiple_times
from .essentials import create_functional_model
from .essentials import use_limit_growth_function
import copy
import numpy as np

def extract_search_space(layers):
    dimensions = []
    param_names = []

    for idx, layer in enumerate(layers):
        for key, val in layer.items():
            if key.endswith("Random") and isinstance(val, dict):
                base = key.replace("Random", "")
                name = f"{base}_{idx}"
                param_names.append(name)
                value_type = val.get("type")
                step = val.get("step", 1)

                if value_type == "numeric" or value_type == "numeric-test":
                    min_val = val["min"]
                    max_val = val["max"]

                    is_float = (
                        isinstance(step, float) or
                        isinstance(min_val, float) or
                        isinstance(max_val, float)
                    )

                    if is_float:
                        dimensions.append(Real(min_val, max_val, name=name))
                    else:
                        dimensions.append(Integer(min_val, max_val, name=name))

                elif val["type"] == "text":
                    dimensions.append(Categorical(val["options"], name=name))

    # nothing to optimize, raise error
    if(len(param_names) == 0):
        raise ValueError("No optimizable parameters found")
    return dimensions, param_names

def inject_params(layers, param_names, values):
    injected = copy.deepcopy(layers)
    for i, name in enumerate(param_names):
        parts = name.split("_")
        base_key = parts[0]
        idx = int(parts[1])
        injected[idx][base_key] = values[i]
    return injected

def bayesian_optimization(layers, settings, x_train, y_train, x_val, y_val, max_trials=15, num_runs=3, threshold=0.7, monitor_metric='val_accuracy'):
    user_id = session.get("user_id")
    dimensions, param_names = extract_search_space(layers)

    best_model = None
    best_score = 0
    best_history = []
    best_params = {}

    epoch_number = 1

    def objective(values):
        nonlocal best_model, best_score, best_history, best_params, epoch_number

        try:
            for i, name in enumerate(param_names):
                parts = name.split("_")
                base_key = parts[0]
                layer_idx = int(parts[1])
                layer = layers[layer_idx]
                rand_key = f"{base_key}Random"
                if rand_key in layer:
                    limited = use_limit_growth_function(
                        layer[rand_key],
                        growth_limiter_manager.get_growth_function(user_id),
                        growth_limiter_manager.get_current_progress(user_id),
                        growth_limiter_manager.get_max_progress(user_id)
                    )
                    task_protocol_manager.get_log(user_id).get_or_create_epoch(epoch_number).limits.append(limited)

            injected_layers = inject_params(layers, param_names, values)
            model, used_params = create_functional_model(injected_layers, settings)

            model, score, history = train_multiple_times(
                model, x_train, y_train, x_val, y_val,
                num_runs=num_runs, threshold=threshold, monitor_metric=monitor_metric,
                epochs=settings["epochs"], batch_size=settings["batch_size"], user_id=user_id
            )

            layers_info = []
            for layer in model.layers:
                neurons = getattr(layer, 'units', getattr(layer, 'filters', None))
                layer_info = {
                    'layer_name': layer.name,
                    'layer_type': layer.__class__.__name__,
                    'num_params': layer.count_params(),
                    'trainable': layer.trainable,
                    'neurons': neurons,
                }
                layers_info.append(layer_info)

            task_protocol_manager.get_log(user_id).get_or_create_epoch(epoch_number)
            task_protocol_manager.get_log(user_id).log_model_to_epoch(
                epoch_number=epoch_number,
                model_id=epoch_number,
                architecture=layers_info,
                history=history,
                parameters=used_params,
                results=score
            )

            progress_manager.update_progress(user_id, (epoch_number / n_calls) * 100)
            growth_limiter_manager.update_progress(user_id)

            if termination_manager.is_terminated(user_id):
                print("\u274c Task terminated by user.")
                raise Exception("Task terminated by user.")
            if time_limit_manager.has_time_exceeded(user_id):
                print("\u23f1\ufe0f Time limit exceeded.")
                raise Exception("Time limit exceeded.")

            if score > best_score:
                best_score = score
                best_model = model
                best_history = history
                best_params = used_params

            epoch_number += 1
            return -score
        except Exception as e:
            print("Bayesian trial error:", e)
            epoch_number += 1
            return 1.0

    n_calls = max(max_trials, 10)
    result = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=42)

    return best_model, best_score, best_history, best_params
