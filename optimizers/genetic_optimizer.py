import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from utils.task_progress_manager import progress_manager, termination_manager
from utils.time_limit_manager import time_limit_manager
from .essentials import create_functional_model, process_parameters
from flask import session
from utils.task_progress_manager import growth_limiter_manager
from utils.task_progress_manager import ExternalTerminationCallback
from utils.task_protocol_manager import task_protocol_manager
import warnings

def train_multiple_times(model, x_train, y_train, x_val, y_val, threshold, num_runs=1, monitor_metric='accuracy', epochs=10, batch_size=32, user_id="", es_patience=10, es_delta=0.01):
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
                print(f"Training run {i+1}")
                history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=batch_size, callbacks=[early_stopping, external_termination], verbose=1)

                # save metric history
                for epoch, value in enumerate(history.history[monitor_metric]):
                    epoch_history.append({"epoch": epoch + 1, "value": round(value, 3)})

                final_metric_value = history.history[monitor_metric][-1]

                if final_metric_value > best_metric_value:
                    best_metric_value = final_metric_value
                    best_weights = model.get_weights()  
                    best_epoch_history = epoch_history

                # skip if not within threshold
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

import random

# create initial population
def initialize_population(layers, settings, population_size):
    try:
        population = []
        for _ in range(population_size):
            model, params = create_functional_model(layers, settings)
            population.append((model, params))
        return population
    except Exception as e:
        raise e

# evaluate fitness of the model
def evaluate_fitness(model, x_train, y_train, x_val, y_val, threshold, num_runs=3, monitor_metric='accuracy',  epochs = 10, batch_size = 32, user_id="", es_patience=10, es_delta=0.01):
    try:
        _, metric_value, metric_history = train_multiple_times(model, x_train, y_train, x_val, y_val, threshold, num_runs=num_runs, monitor_metric=monitor_metric, epochs=epochs, batch_size=batch_size, user_id=user_id, es_patience=es_patience, es_delta=es_delta)
        return metric_value, metric_history
    except Exception as e:
        raise e
    
# parent selection
def select_parents(population, fitness_scores, num_parents, selection_method="Roulette"):
    try:
        if selection_method == "Roulette":
            total_fitness = sum(fitness_scores)
            probabilities = [score / total_fitness for score in fitness_scores]
            selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities, replace=False)
            return [population[i] for i in selected_indices]

        elif selection_method == "Tournament":
            # Tournament Selection
            selected = []
            for _ in range(num_parents):
                competitors = random.sample(list(zip(population, fitness_scores)), k=3) 
                best_competitor = max(competitors, key=lambda x: x[1])
                selected.append(best_competitor[0])
            return selected

        elif selection_method == "Rank":
            # Rank-Based Selection
            sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            ranks = range(1, len(sorted_population) + 1)
            total_rank = sum(ranks)
            probabilities = [rank / total_rank for rank in ranks]
            selected_indices = np.random.choice(len(sorted_population), size=num_parents, p=probabilities, replace=False)
            return [sorted_population[i][0] for i in selected_indices]

        elif selection_method == "Random":
            # Random Selection
            return random.sample(population, num_parents)

        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
    except Exception as e:
        raise e    


# create a child from two parents
def crossover(parents = [], method="classic"):
    try:
        if not parents or len(parents) < 2:
            raise ValueError("You need at least 2 parents")
        print("parent1,", parents[0])
        print("parent2,",parents[1])
        child = []

        #classic 2 parent one point crossover
        if method=="classic":
            for l in range(len(parents[0])):

                #do generator crossover
                if l == 2:
                    if len(parents[0][2]) == 0:
                        child.append([])
                        continue
                    g_t_c=0
                    c = gen_crossover(parents[0][2][g_t_c], parents[1][2][g_t_c])
                    child.append(c)
                    continue
                i = parents[0][l]
                break_point = random.randint(0, len(i))
                par1 = list(parents[0][l].items())[:break_point]
                par2 = list(parents[1][l].items())[break_point:]
                child.append(dict(par1 + par2))
        #possible more methods

        return child
    except Exception as e:
        raise e

def gen_crossover(par1, par2, method="onePoint"):
    try:
        parent1 = par1
        print("gen_crossover_par1", par1)
        parent2 = par2

        up1 = parent1['used_parameters']
        up2 = parent2['used_parameters']
        ls1 = parent1['layers_sequence']
        ls2 = parent2['layers_sequence']

        child_up = []
        child_ls = []

        #creates one point from which the two parents split (if the point is more, than one parents length, the whole parent will
        #be used and the other one will get connected on top of that)
        if method=="onePoint":
            max_length = max(len(ls1), len(ls2))
            cp = random.randint(0, max_length - 1)

            child_up = up1[:cp] + up2[cp:]
            child_ls = ls1[:cp] + ls2[cp:]


        #TBA - possible method that does crossover based on layerId rather than fixed point
        elif method=="layerId":
            pass
        
        
        #used_rules and used_layers should be the same, in case this proves wrong it could be edited to combine them
        child = {
            'used_parameters': child_up,
            'layers_sequence': child_ls,
            'used_rules': parent1['used_rules'],
            'used_layers': parent1['used_layers']
        }

        return [child]
    except Exception as e:
        raise e

def mutate(params, layers, settings, mutation_rate=0.1, method="onePoint"):
    try:
        mutated_params = params.copy()

        if method == "onePoint":

            if random.random() > mutation_rate:
                _, pp_l = process_parameters(layers)
                _, pp_s = process_parameters([settings])
                new=[pp_l, pp_s]
                mutation_list = 0

                #which parameter mutated
                if len(params[mutation_list]) == 0:
                    return mutated_params
                else:
                    mutation_index = random.randint(0, len(params[mutation_list])-1)

                x= list(mutated_params[mutation_list].items())
                x[mutation_index] = (x[mutation_index][0], list(new[mutation_list].items())[mutation_index][1])
                mutated_params[mutation_list] = dict(x)

        return mutated_params
    except Exception as e:
        raise e

def genetic_optimization(
    layers, 
    settings, 
    x_train, 
    y_train, 
    x_val, 
    y_val, 
    population_size=10, 
    num_generations=5, 
    num_of_additions=1,
    num_parents=4, 
    mutation_rate=0.1, 
    num_runs=3, 
    monitor_metric='accuracy',
    selection_method="Roulette", 
    threshold=0.7, 
    max_models=5,
    trackProgress = True,
    user_id="", 
    es_patience=10, 
    es_delta=0.01
):
    try:
        # Inicializate populaction
        print("user_id v gen ", user_id)
        population = initialize_population(layers, settings, population_size)
        fitness_results = [
            evaluate_fitness(model, x_train, y_train, x_val, y_val, threshold, num_runs, monitor_metric, epochs=settings["epochs"], batch_size=settings["batch_size"], user_id=user_id, es_patience=es_patience, es_delta=es_delta)
            for model, _ in population
        ]
        fitness_scores = [metric_value for metric_value, _ in fitness_results]
        fitness_histories = [best_history for _, best_history in fitness_results]

        global_best_score = max(fitness_scores)
        global_best_index = fitness_scores.index(global_best_score)
        global_best_model, global_best_params = population[global_best_index]
        global_best_history = fitness_histories[global_best_index]

        user_id = session.get("user_id")

        for generation in range(num_generations):
            task_protocol_manager.get_log(user_id).get_or_create_epoch(epoch_number = (generation+1))

            parents = select_parents(population, fitness_scores, num_parents, selection_method)

            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(parents, 2)

                #there is literaly no point in using genetic optimization with no hyperparameters or generator, but in case that happens ...
                if len(parent1[1][0])+len(parent1[1][1])+len(parent1[1][2])==0:
                            while len(new_population) < population_size:
                                model, params = create_functional_model(layers, settings)
                                new_population.append((model, params))
                            break
                child_params = crossover([parent1[1], parent2[1]])
                child_params = mutate(child_params, layers, settings, mutation_rate)
                child_model, _ = create_functional_model(layers, settings, child_params)
                new_population.append((child_model, child_params))

            population = new_population

            # add some extra models to keep population fresh 
            for i in range(num_of_additions):
                 model, params = create_functional_model(layers, settings)
                 population.append((model, params))            
            
            fitness_results = [
                evaluate_fitness(model, x_train, y_train, x_val, y_val, threshold, num_runs, monitor_metric, epochs=settings["epochs"], batch_size=settings["batch_size"], user_id=user_id, es_patience=es_patience, es_delta=es_delta)
                for model, _ in population
            ]
            # protocol models in generation
            for model_index, ((model, used_params), (metric_value, history)) in enumerate(zip(population, fitness_results)):
                layers_info = []
                for layer in model.layers:
                    layer_info = {
                        'layer_name': layer.name,
                        'layer_type': layer.__class__.__name__,
                        'num_params': layer.count_params(),
                        'trainable': layer.trainable
                    }
                    try:
                        layer_info['output_shape'] = str(layer.output_shape)
                    except AttributeError:
                        layer_info['output_shape'] = "N/A"
                    layers_info.append(layer_info)

                task_protocol_manager.get_log(user_id).log_model_to_epoch(
                    epoch_number=generation + 1,
                    model_id=f"model_{generation + 1}_{model_index + 1}",
                    architecture=layers_info,
                    parameters=used_params,
                    history=history,
                    results=metric_value
                )

            fitness_scores = [metric_value for metric_value, _ in fitness_results]
            fitness_histories = [best_history for _, best_history in fitness_results]

            generation_best_score = max(fitness_scores)
            generation_best_index = fitness_scores.index(generation_best_score)
            generation_best_model, generation_best_params = population[generation_best_index]
            generation_best_history = fitness_histories[generation_best_index]

            if generation_best_score > global_best_score:
                global_best_score = generation_best_score
                global_best_model = generation_best_model
                global_best_params = generation_best_params
                global_best_history = generation_best_history
            
            if(trackProgress):
                progress = ((generation + 1) / num_generations) * 100 
                progress_manager.update_progress(user_id, progress)

            # print(f"Best model in generation {generation + 1}: {monitor_metric} = {generation_best_score}")
            if(termination_manager.is_terminated(user_id)):
                task_protocol_manager.log_item(user_id, "stopped_by", "user")
                raise Exception("Task terminated by user")
            if(time_limit_manager.has_time_exceeded(user_id)):
                task_protocol_manager.log_item(user_id, "stopped_by", "timeout")
                break
            growth_limiter_manager.update_progress(user_id)


        best_model_index = fitness_scores.index(max(fitness_scores))
        best_model, best_params = population[best_model_index]

        return global_best_model, global_best_score, global_best_history, global_best_params
    except Exception as e:
        raise e