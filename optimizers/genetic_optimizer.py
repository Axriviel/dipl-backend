import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from .essentials import create_functional_model, process_parameters

# Funkce pro více tréninků jednoho modelu
def train_multiple_times(model, x_train, y_train, x_val, y_val, num_runs=3, threshold=0.7, monitor_metric='val_accuracy'):
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=5, min_delta=0.01, mode='max', restore_best_weights=True)
    epoch_history = []
    best_metric_value = 0
    best_weights = None  # Uchová váhy modelu s nejlepší finální hodnotou metriky

    for i in range(num_runs):
        try:
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
    return model, best_metric_value, epoch_history

import random

# Funkce pro vytvoření počáteční populace modelů
def initialize_population(layers, settings, population_size):
    population = []
    for _ in range(population_size):
        model, params = create_functional_model(layers, settings)
        population.append((model, params))
    return population

# Fitness funkce pro hodnocení modelů
def evaluate_fitness(model, x_train, y_train, x_val, y_val, num_runs=3, monitor_metric='val_accuracy'):
    _, metric_value, metric_history = train_multiple_times(model, x_train, y_train, x_val, y_val, num_runs=num_runs, monitor_metric=monitor_metric)
    return metric_value, metric_history

# Výběr nejlepších rodičů
def select_parents(population, fitness_scores, num_parents, selection_method="Roulette"):
    """
    Výběr rodičů podle zvolené metody selekce.
    """
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


# Křížení rodičů pro generování potomků
#fixed
def crossover(parents = [], method="classic"):
    if not parents or len(parents) < 2:
        raise ValueError("You need at least 2 parents")
    print(parents[0])
    print(parents[1])
    child = []
    
    #classic 2 parent one point crossover
    if method=="classic":
        l = 0
        for i in parents[0]:
            break_point = random.randint(0, len(i))
            par1 = list(parents[0][l].items())[:break_point]
            par2 = list(parents[1][l].items())[break_point:]
            child.append(dict(par1+par2))
            l += 1
    
    #possible more methods
    #vícebodové křížení, brát každý jeden parametr náhodně z náhodného rodiče ...
    #možnost křížení z více rodičů ...
    if method =="test":
        pass
    print(child)
    return child

# Mutace parametrů modelu
def mutate(params, layers, settings, mutation_rate=0.1, method="onePoint"):
    mutated_params = params.copy()
    print("mutated parametry jsou:", mutated_params)
    
    if method == "onePoint":
        
        #tuhle logiku dát asi ven, když už budu volat mutate, tak na parametry, které určitě zmutují
        if random.random() > mutation_rate:
            _, pp_l = process_parameters(layers)
            _, pp_s = process_parameters([settings])
            new=[pp_l, pp_s]
            #which list mutated
#             mutation_list = random.randint(0, len(params)-1)
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
    selection_method="Roulette"  # Metoda výběru rodičů
):
    # Inicializace populace
    population = initialize_population(layers, settings, population_size)
    fitness_results = [
        evaluate_fitness(model, x_train, y_train, x_val, y_val, num_runs, monitor_metric)
        for model, _ in population
    ]
    fitness_scores = [metric_value for metric_value, _ in fitness_results]
    fitness_histories = [best_history for _, best_history in fitness_results]

    # Globální nejlepší model
    global_best_score = max(fitness_scores)
    global_best_index = fitness_scores.index(global_best_score)
    global_best_model, global_best_params = population[global_best_index]
    global_best_history = fitness_histories[global_best_index]


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
            evaluate_fitness(model, x_train, y_train, x_val, y_val, num_runs, monitor_metric)
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
        
        print(f"Best model in generation {generation + 1}: {monitor_metric} = {generation_best_score}")


    # Vrácení nejlepšího modelu a jeho parametrů
    best_model_index = fitness_scores.index(max(fitness_scores))
    best_model, best_params = population[best_model_index]

    #best_model_history = []

    #return best_model, max(fitness_scores), best_model_history, best_params
    return global_best_model, global_best_score, global_best_history, global_best_params