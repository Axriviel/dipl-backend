from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D
from tensorflow.keras.models import Model
import random
from utils.dataset_storing import load_dataset
from sklearn.model_selection import train_test_split

#
def create_optimized_model(layers, settings, dataset_path):
    
    opt_method = settings["opt_algorithm"]
    

    if opt_method == "random":
        from optimizers.random_optimizer import random_search

        x_train, x_test, y_train, y_test = process_dataset(dataset_path)
        print("dataset processed")

        # Spuštění random search, můžeme zvolit například sledování metriky val_loss
        b_model, b_metric_val, b_metric_history = random_search(layers, settings, 
                                                  x_train=x_train, y_train=y_train, 
                                                  x_val=x_test, y_val=y_test, 
                                                  num_models=5, num_runs=3, 
                                                  threshold=0.7, 
#                                                   monitor_metric='val_loss')
                                                  monitor_metric=settings["monitor_metric"])
        
        print(f"Best model found with {settings['monitor_metric']} : {b_metric_val}")
        b_model.summary()
        return b_model, b_metric_val, b_metric_history

    elif opt_method == "genetic":
        raise Exception("Not yet supported")
    
    return None

#přidat podmínečné zpracování - část na FE a tady asi taky podle toho, co to je za dataset (csv, nebo něco jiného?), stejně tak něco s tím y
def process_dataset(dataset_path):
    dataset = load_dataset(dataset_path)
    x = dataset.iloc[:,0:8] 
    y = dataset.iloc[:,8]
    # print(x)
    # print(y)
    return train_test_split(x, y, test_size=0.2) 

def generate_random_value(random_info):
    """Generate a random value based on the provided randomization info."""
    if random_info['type'] == 'numeric':
        return random.randint(random_info['min'], random_info['max'])
    elif random_info['type'] == 'text':
        return random.choice(random_info['options'])
    return None

def process_layer_params(layer):
    """Zpracuje parametry vrstvy, odstraní klíče obsahující 'Random' a uloží náhodné hodnoty pod odpovídající klíč."""
    processed_params = {}
    
    for key, value in layer.items():
        # Kontrola klíčů, které obsahují "Random", a uložení náhodné hodnoty pod klíč bez "Random"
        if 'Random' in key:
            base_key = key.replace('Random', '')  # Získání základního klíče (např. 'units' z 'unitsRandom')
            processed_params[base_key] = generate_random_value(value)
        elif key in ['id', 'inputs']:  # Vynechání klíčů, které nejsou platné pro vytváření vrstev
            continue
        else:
            processed_params[key] = value
    
    return processed_params

def get_layer(layer):
    lp = process_layer_params(layer)
    lt = lp["type"].lower()
    lp.pop("type")
    print(lt)
    print(lp)
    
    layer_switch = {
    'input': Input,
    'dense': Dense,
    'conv2d': Conv2D,
    # Můžeš přidat další vrstvy podle potřeby
    }
    
        # Zkontrolujeme, zda je daný typ vrstvy podporován
    if lt in layer_switch:
        layer_class = layer_switch[lt]  # Získáme třídu vrstvy z našeho slovníku

        if layer_class == "generator": #return instance of generator with required configuration
            pass
        
        return layer_class(**lp)  # Vytvoříme instanci vrstvy s parametry
    else:
        raise ValueError(f"Unsupported layer type: {lt}")
        


def create_functional_model(layers, settings):
    # Uložíme si výstupy jednotlivých vrstev podle jejich id
    layer_outputs = {}

    # 1. Nejprve vytvoříme vstupní vrstvu
    input_layer = get_layer(layers[0])
    layer_outputs[layers[0]['id']] = input_layer
    
    # 2. Pro každou další vrstvu zpracujeme vstupy a propojíme vrstvy
    for layer in layers[1:]:
#         layer_params = process_layer_params(layer) #zpracujeme parametry
#         layer_type = layer_params.pop('type') #uložíme si layer_type
        
        # Získáme vstupy pro aktuální vrstvu
        input_tensors = [layer_outputs[input_id] for input_id in layer['inputs']]
        
        # Pokud je pouze jeden vstup, použijeme jej přímo, jinak musíme sloučit (např. Concatenate)
        if len(input_tensors) == 1:
            input_tensor = input_tensors[0]
        else:
            # Zde bys mohl použít například Concatenate() pro více vstupů
            input_tensor = Concatenate()(input_tensors)
            #raise ValueError("Multiple inputs not supported yet")
        
        # Vytvoříme vrstvu a propojíme ji se vstupy
        new_layer = get_layer(layer)
        output_tensor = new_layer(input_tensor)
        
        # Uložíme výstup nové vrstvy pod její id
        layer_outputs[layer['id']] = output_tensor
        
#tohle udělat na frontendu - uživatel si prostě tu výstupní vrstvu musí zadat sám.
        #     # 3. Přidání výstupní vrstvy podle typu problému
#     if problem_type == 'binary_classification':
#         output_tensor = Dense(1, activation='sigmoid')(output_tensor)
#         loss = 'binary_crossentropy'
        
#     elif problem_type == 'multiclass_classification':
#         output_tensor = Dense(n_classes, activation='softmax')(output_tensor)
#         loss = 'categorical_crossentropy'
#     else:
#         raise ValueError(f"Unknown problem type: {problem_type}")

    # 4. Model specifikuje vstupy a výstupy (poslední vrstva bude výstupní)
    model = Model(inputs=input_layer, outputs=output_tensor)
    
    #zadávání metriky se dělá v compile
    model.compile(
        optimizer=settings['optimizer'], 
        loss=settings['loss'], 
        metrics=settings['metrics']
    )
    
    return model