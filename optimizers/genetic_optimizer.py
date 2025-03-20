import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from utils.task_progress_manager import progress_manager
from .essentials import create_functional_model, process_parameters
from flask import session

# Funkce pro více tréninků jednoho modelu
def train_multiple_times(model, x_train, y_train, x_val, y_val, threshold, num_runs=3, monitor_metric='val_accuracy'):
    try:
        early_stopping = EarlyStopping(monitor=monitor_metric, patience=5, min_delta=0.01, mode='max', restore_best_weights=True)
        best_epoch_history = []
        best_metric_value = 0
        best_weights = None  # Uchová váhy modelu s nejlepší finální hodnotou metriky

        for i in range(num_runs):
            try:
                epoch_history = []
                print(f"Training run {i+1}")
                history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

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
    except Exception as e:
        raise e

import random

# Funkce pro vytvoření počáteční populace modelů
def initialize_population(layers, settings, population_size):
    try:
        population = []
        for _ in range(population_size):
            model, params = create_functional_model(layers, settings)
            population.append((model, params))
        return population
    except Exception as e:
        raise e

# Fitness funkce pro hodnocení modelů
def evaluate_fitness(model, x_train, y_train, x_val, y_val, threshold, num_runs=3, monitor_metric='val_accuracy',  ):
    try:
        _, metric_value, metric_history = train_multiple_times(model, x_train, y_train, x_val, y_val, threshold, num_runs=num_runs, monitor_metric=monitor_metric)
        return metric_value, metric_history
    except Exception as e:
        raise e
    
# Výběr nejlepších rodičů
def select_parents(population, fitness_scores, num_parents, selection_method="Roulette"):
    try:
        if selection_method == "Roulette":
            # Roulette Wheel Selection (Proporční výběr)
            total_fitness = sum(fitness_scores)
            probabilities = [score / total_fitness for score in fitness_scores]
            selected_indices = np.random.choice(len(population), size=num_parents, p=probabilities, replace=False)
            return [population[i] for i in selected_indices]

        elif selection_method == "Tournament":
            # Tournament Selection
            selected = []
            for _ in range(num_parents):
                competitors = random.sample(list(zip(population, fitness_scores)), k=3)  # Turnaj o 3 jedince
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


# Křížení rodičů pro generování potomků
#fixed
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
                        # print("skipuju gen")
                        child.append([])
                        continue
                    # print("jdu na gen_c")
                    #generator to crossover - only matters in case of using multiple generators
                    g_t_c=0
                    #list of dictionaries for generator replication
                    c = gen_crossover(parents[0][2][g_t_c], parents[1][2][g_t_c])
                    child.append(c)
                    continue
                i = parents[0][l]
                break_point = random.randint(0, len(i))
                # print(break_point)
                par1 = list(parents[0][l].items())[:break_point]
                # print(par1)
                par2 = list(parents[1][l].items())[break_point:]
                # print(par2)
                child.append(dict(par1 + par2))


        #possible more methods
        #vícebodové křížení, brát každý jeden parametr náhodně z náhodného rodiče ...
        #možnost křížení z více rodičů ...
        if method =="test":
            pass
        print("child je,",child)

        return child
    except Exception as e:
        raise e

def gen_crossover(par1, par2, method="onePoint"):
    try:
        # p1_third a p2_third jsou listy o jednom prvku - např. par1[2], par2[2]
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


        #TBA - method that does crossover based on layerId rather than fixed point
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

# Mutace parametrů modelu
def mutate(params, layers, settings, mutation_rate=0.1, method="onePoint"):
    try:
        mutated_params = params.copy()
        print("mutated parametry jsou:", mutated_params)

        if method == "onePoint":

            #tuhle logiku dát asi ven, když už budu volat mutate, tak na parametry, které určitě zmutují
            if random.random() > mutation_rate:
                _, pp_l = process_parameters(layers)
                _, pp_s = process_parameters([settings])
                new=[pp_l, pp_s]
                #which list mutated
#                 mutation_list = random.randint(0, len(params)-1)
                mutation_list = 0

                #možná přidat něco jako když je prázdný, použít jiný list ...

                #which parameter mutated
                if len(params[mutation_list]) == 0:
                    return mutated_params
                else:
                    mutation_index = random.randint(0, len(params[mutation_list])-1)
                print(mutation_list)
                print(mutation_index)

                print(new)

                x= list(mutated_params[mutation_list].items())
                x[mutation_index] = (x[mutation_index][0], list(new[mutation_list].items())[mutation_index][1])
                mutated_params[mutation_list] = dict(x)




        return mutated_params
    except Exception as e:
        raise e

# Hlavní genetická optimalizace
def genetic_optimization(
    layers, 
    settings, 
    x_train, 
    y_train, 
    x_val, 
    y_val, 
    population_size=10, 
    num_generations=5, 
    num_parents=4, 
    mutation_rate=0.1, 
    num_runs=3, 
    monitor_metric='val_accuracy',
    selection_method="Roulette",  # Metoda výběru rodičů
    threshold=0.7, 
):
    try:
        # Inicializace populace
        population = initialize_population(layers, settings, population_size)
        fitness_results = [
            evaluate_fitness(model, x_train, y_train, x_val, y_val, threshold, num_runs, monitor_metric)
            for model, _ in population
        ]
        fitness_scores = [metric_value for metric_value, _ in fitness_results]
        fitness_histories = [best_history for _, best_history in fitness_results]

        # Globální nejlepší model
        global_best_score = max(fitness_scores)
        global_best_index = fitness_scores.index(global_best_score)
        global_best_model, global_best_params = population[global_best_index]
        global_best_history = fitness_histories[global_best_index]

        user_id = session.get("user_id")

        for generation in range(num_generations):
            print(f"Generation {generation + 1}")
            # Výběr rodičů
            parents = select_parents(population, fitness_scores, num_parents, selection_method)
            print("rodice jsou", parents)

            # Generování nové populace pomocí křížení a mutace
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(parents, 2)
                print("prent1", parent1)
                print("prent1", parent1[1])

                #there is literaly no point in using genetic optimization with no hyperparameters or generator, but in case that happens ...
                if len(parent1[1][0])+len(parent1[1][1])+len(parent1[1][2])==0:
                            while len(new_population) < population_size:
                                model, params = create_functional_model(layers, settings)
                                new_population.append((model, params))
                            break
                print("jdeme na crossover")
                child_params = crossover([parent1[1], parent2[1]])
                print("child params1", child_params)
                child_params = mutate(child_params, layers, settings, mutation_rate)
                print("child params", child_params)
                child_model, _ = create_functional_model(layers, settings, child_params)
                new_population.append((child_model, child_params))
                print("nova populace", new_population)

            # Aktualizace populace a fitness score
            population = new_population
            fitness_results = [
                evaluate_fitness(model, x_train, y_train, x_val, y_val, threshold, num_runs, monitor_metric)
                for model, _ in population
            ]
            fitness_scores = [metric_value for metric_value, _ in fitness_results]
            fitness_histories = [best_history for _, best_history in fitness_results]
            # Nejlepší model generace
            generation_best_score = max(fitness_scores)
            generation_best_index = fitness_scores.index(generation_best_score)
            generation_best_model, generation_best_params = population[generation_best_index]
            generation_best_history = fitness_histories[generation_best_index]

            # Aktualizace globálního nejlepšího modelu
            if generation_best_score > global_best_score:
                global_best_score = generation_best_score
                global_best_model = generation_best_model
                global_best_params = generation_best_params
                global_best_history = generation_best_history

            progress = ((generation + 1) / num_generations) * 100  # Progress jako % dokončení
            progress_manager.update_progress(user_id, progress)

            print(f"Best model in generation {generation + 1}: {monitor_metric} = {generation_best_score}")


        # Vrácení nejlepšího modelu a jeho parametrů
        best_model_index = fitness_scores.index(max(fitness_scores))
        best_model, best_params = population[best_model_index]

        #best_model_history = []

        #return best_model, max(fitness_scores), best_model_history, best_params
        return global_best_model, global_best_score, global_best_history, global_best_params
    except Exception as e:
        raise e