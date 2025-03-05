from models.model import Model
from tensorflow.keras.callbacks import EarlyStopping
from .essentials import create_functional_model
from flask import session
from utils.task_progress_manager import progress_manager


#counts match score between required tags (user defined task) and target tags (models in database)
#numbers may need to be adjusted
def get_tag_score(source_tags, target_tags):
    match_score = 0
    print(type(source_tags))
    print(type(target_tags))

    
    if source_tags["task"] == target_tags["task"]:
        match_score += 30

    
    if source_tags["dataset"] == target_tags["dataset"]:
        match_score += 50
        
    if source_tags["metric"] == target_tags["metric"]:
        match_score += 20
    
    for tag in source_tags["userTags"]:
        if tag in target_tags["userTags"]:
            match_score += 10
            
    return match_score

# 
def find_candidates(source_tags, task, num_of_models = 3):
    import json
    source_tags = json.loads(source_tags)

    candidates = Model.query.filter_by(used_task=task).all()
    print("candidates form db:", candidates)
    #if no candidates found, try taking all models
    if not candidates:
        candidates = Model.query.all()

    #if candidates dont exist even after query all return empty
    if not candidates:
        return []
        # Seznam pro ukládání skóre modelů
    scored_models = []

    # Porovnání každého modelu s tagy
    for model in candidates:
        # Extrahuj informace z databáze
        target_tags = json.loads(model.used_tags)
        print(target_tags, "targ tagy")
        print(source_tags, "src tagy")

        # Získej skóre pomocí get_tag_score
        score = get_tag_score(source_tags, target_tags)
        print("score:", score)

        # Přidej model a jeho skóre do seznamu
        scored_models.append({
            "id": model.id,
            # "model_name": model.model_name,
            "config": model.creation_config,
            "params": model.used_params,
            "user_owner": model.user_id,
            "score": score,
            "metric_value": model.metric_value,
            # "watched_metric": model.watched_metric
        })
        print("append proveden")
    print(scored_models, "modely")

    # Seřaď modely podle skóre (sestupně) a preferovat lepší metric_value
    scored_models.sort(key=lambda x: (x["score"], x["metric_value"]), reverse=True)
    print("sort proveden")
    return scored_models[:num_of_models]

def tagging_optimization(layers, 
    settings, 
    x_train, 
    y_train, 
    x_test, 
    y_test,
    num_of_models, 
    num_runs, 
    threshold, 
    opt_data
    ):
    potential_models = find_candidates(opt_data["tags"], opt_data["task_type"])
    print("potential_models-tagging", potential_models)

    user_id = session.get("user_id")
    progress_manager.update_progress(user_id, 0)

    found_models = []

    #tba - retrain potential models and evaluate
    if len(potential_models) != 0:
        for index, m in enumerate(potential_models):
        #načíst model (načíst config z databáze a ten použít, v něm nahradit výstupní vrstvu a vstupní dimenzi v inputu)
        #možná načíst váhy z existujícího modelu a ty do toho nového nahrát?
            # print(m, "potential model")
            
            #config saves [layers, settings, dataset_config], thats why we need only 1st item
            saved_model_layers = m["config"][0]
            saved_model_params = m["params"]

            #modify input_shape of existing config to new one
            saved_model_layers[0]["shape"]=layers[0]["shape"]
            #replace last layer with new one
            saved_model_layers[-1]=layers[-1]

            print(saved_model_layers, "layers nove")
            print(saved_model_params, "params nove")

            model, used_params = create_functional_model(saved_model_layers, settings, params=saved_model_params)
            print("model created")
            trained_model, metric_value, metric_history = train_multiple_times(
            model, x_train, y_train, x_test, y_test, num_runs=num_runs, threshold=threshold, monitor_metric=settings["monitor_metric"], epochs=settings["epochs"], batch_size=settings["batch_size"])
            print("model_trained")
            
            found_models.append([trained_model, metric_value, metric_history, used_params])
        print("Celkem nalezeno: ", len(found_models), "modelů")

    progress_manager.update_progress(user_id, 50)            

    # print("nyní budeme generovat")
    #pokud je modelů málo, nebo stejně chceme vytvořit několik vlastních ...
    #stejně tak by zde bylo možno použít genetický přístup ...
    from .random_optimizer import random_search
    for i in range(num_of_models):
        b_model, b_metric_val, b_metric_history, used_params = random_search(layers, settings, 
                                                  x_train=x_train, y_train=y_train, 
                                                  x_val=x_test, y_val=y_test, 
                                                  num_models=5, num_runs=3, 
                                                  threshold=0.7, 
                                                  monitor_metric=settings["monitor_metric"])
                
        found_models.append([b_model, b_metric_val, b_metric_history, used_params])

    sorted_models = sorted(found_models, key=lambda x: x[1], reverse=True)
    print("sorted_models: ",sorted_models)
    
    
    progress_manager.update_progress(user_id, 100)
    b_model = sorted_models[0]
    print(b_model)
    best_model = b_model[0]
    best_metric_value = b_model[1]
    best_metric_history = b_model[2]
    best_model_params = b_model[3]
    # progress_manager.update_progress(user_id, 100)
    return best_model, best_metric_value, best_metric_history, best_model_params





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