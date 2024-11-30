import json
import nni
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, LSTM, Concatenate
# from optimizers.essentials import create_functional_model
import numpy as np
import random
import nni
import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def load_dataset(path):
    try:
        if path.endswith('.csv'):
            dataset = pd.read_csv(path)
        elif path.endswith('.tsv'):
            dataset = pd.read_csv(path, delimiter='\t')
        elif path.endswith('.npy'):
            # NumPy pole převedeme na pandas DataFrame, každá hodnota bude v samostatném sloupci
            np_data = np.load(path)
            dataset = pd.DataFrame(np_data, columns=[f"Column_{i}" for i in range(np_data.shape[1])])
        elif path.endswith('.npz'):
            # Pro NPZ soubory obsahující více polí můžeme načíst první pole nebo všechna pole
            np_data = np.load(path)
            dataset = pd.DataFrame({k: v for k, v in np_data.items()})
        elif path.endswith('.h5'):
            dataset = pd.read_hdf(path)
        return dataset
    except Exception as e:
        print("Error in loading dataset" + e)
        raise

class Generator:
    def __init__(self, attempts=5):
        self.rules = {}
        self.attempts = attempts

    def setRules(self, layers):
        """Nastaví pravidla pro vrstvy podle možných následných vrstev."""
        print("setting rules:", layers)
        rules = {}
        for layer in layers:
            layer_id = layer["id"]
            # Uložení typu vrstvy a zpracovaných parametrů

            if len(layers) == 1:
                layer["inputs"] = layer["id"]

            possible_next_layers = [
                (l["type"], l["id"]) for l in layers if l["id"] in layer.get("inputs", [])
            ]
            
            print("pos_layers", possible_next_layers)

            rules[layer_id] = {
                "layer": layer,
                "params": layer,
                "next_layers": possible_next_layers
            }
            print("rules", rules)
        self.rules = rules

    def generateFunc(self, size, inp, out=None, firstLayer=None):
        """Generuje strukturu neuronové sítě s ohledem na specifikovaná pravidla."""
        inpStruct = inp
        attempts = self.attempts

        used_parameters = []

        while attempts > 0:
            try:
                struct = inpStruct
                current_layer_id = firstLayer if firstLayer else list(self.rules.keys())[0]
                
                for i in range(size):
                    rule = self.rules.get(current_layer_id, None)
                    if rule is None:
                        raise ValueError(f"Layer with ID {current_layer_id} not found in rules.")
                    
                    print("used rule:", rule["layer"])
                    processed_rule_layer, used_rule_param = process_parameters([rule["layer"]])
                    used_parameters.append(used_rule_param)
                    print("processed_rule_layer", processed_rule_layer)

                    # Použití get_layer pro vytvoření vrstvy se zpracovanými parametry
                    # new_layer = get_layer(rule["layer"], struct)
                    new_layer = get_layer(remove_outer_list(processed_rule_layer), struct)
                    struct = new_layer(struct) if callable(new_layer) else new_layer
                    # print("Vrstva přidána s parametry:", rule["params"])
                    print("Vrstva přidána s parametry:", used_rule_param)

                    # Výběr další vrstvy z možných následujících vrstev
                    if rule["next_layers"]:
                        next_layer_id = random.choice([layer_id for _, layer_id in rule["next_layers"]])
                        current_layer_id = next_layer_id
                    else:
                        break

                if out is not None:
                    return out(struct)
                else:
                    print("toto je struktura:")
                    print(struct)
                    print("used_generator_params", used_parameters)

                    return struct

            except Exception as e:
                print(f"Error during generation: {e}")
                attempts -= 1

        print("Nepodařilo se vygenerovat funkční strukturu v zadaném počtu pokusů.")
        return inpStruct



def create_functional_model(layers, settings, params = None):
    #sending as list so that the change is written here
    print("parametry", params)
    #print("parametr test 0", list(params.values())[0])
    # Uložíme výstupy jednotlivých vrstev podle jejich id
    layer_outputs = {}

    #move away

    #process parameters of layers and models settings
    #tady bude třeba ještě trochu upravit params, bude třeba rozdělit nějak/sjednotit zpracování pro settings a layers
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    processed_layers, used_layers_params = process_parameters(layers, params = params)
    print("params processed")
    processed_settings, used_settings_params = process_parameters([settings])
    #remove outer list
    processed_settings = processed_settings[0]
    print("settings processed")

    unresolved_layers = processed_layers.copy()  # Seznam vrstev, které je třeba ještě zpracovat


    # Prvotní zpracování vstupních vrstev
    while unresolved_layers:
        for layer in unresolved_layers[:]:  # Pro každou nezpracovanou vrstvu
            
            # Pokud je to vstupní vrstva, vytvoříme ji ihned
            if layer["type"] == "Input" and layer["id"] not in layer_outputs:
                input_layer = get_layer(layer, model=None)
                layer_outputs[layer['id']] = input_layer
                unresolved_layers.remove(layer)
                continue
            # Kontrola, zda všechny vstupy pro tuto vrstvu už byly vytvořeny
            if all(input_id in layer_outputs for input_id in layer['inputs']):
                # Vstupy jsou dostupné, můžeme vytvořit tuto vrstvu
                input_tensors = [layer_outputs[input_id] for input_id in layer['inputs']]
                
                # Pokud je pouze jeden vstup, použijeme jej přímo, jinak sloučíme (např. Concatenate)
                input_tensor = input_tensors[0] if len(input_tensors) == 1 else Concatenate()(input_tensors)

                # Zpracování vrstvy podle typu
                if layer["type"] == "Generator":
                    print("Generování vrstvy pomocí generatoru")
                    #generator return whole model, so we need to treat it as such
                    output_tensor = get_layer(layer, input_tensor)
                else:
                    new_layer = get_layer(layer, input_tensor)
                    output_tensor = new_layer(input_tensor)
                
                # Uložíme výstup nové vrstvy pod její id
                layer_outputs[layer['id']] = output_tensor
                unresolved_layers.remove(layer)  # Odstraníme vrstvu z nezpracovaných

    # Specifikace modelu s použitím vstupní vrstvy a poslední vrstvy jako výstupu
    input_layer = layer_outputs[processed_layers[0]['id']]
    output_tensor = list(layer_outputs.values())[-1]  # Poslední vrstva bude výstupní
    
    model = Model(inputs=input_layer, outputs=output_tensor)
    # Kompilace modelu s parametry z nastavení
    model.compile(
        optimizer=processed_settings['optimizer'], 
        loss=processed_settings['loss'], 
        metrics=processed_settings['metrics']
    )
    
    print("used params: ", used_layers_params, used_settings_params)
    return model


def remove_outer_list(x):
    return x[0]

def generate_random_value(random_info):
    """Generate a random value based on the provided randomization info."""
    if random_info['type'] == 'numeric':
        return random.randint(random_info['min'], random_info['max'])
    elif random_info['type'] == 'text':
        return random.choice(random_info['options'])
    return None

def process_parameters(config, params=None, keras_int_params=None):
    import copy
    processed_config = copy.deepcopy(config)
    used_params = {}  #save used params
    paramNum = 0

    if keras_int_params is None:
        keras_int_params = ['units', 'filters', 'kernel_size', 'strides', 'pool_size'] 

    
    for i in processed_config:
        print(i)
        for key, value in i.items():
            print(key, value)
        # Kontrola klíčů, které obsahují "Random", a uložení náhodné hodnoty pod klíč bez "Random"
            if 'Random' in key:
                base_key = key.replace('Random', '')  # Získání základního klíče (např. 'units' z 'unitsRandom')
            
                if params and paramNum < len(params):
                    param_value = list(params.values())[paramNum]

                # convert to int when necessary, becouse quniform returns float every time
                    if base_key in keras_int_params:
                        param_value = int(param_value)
                    i[base_key] = param_value
                else:
                    param_value = generate_random_value(value)
                    i[base_key] = param_value
                    
                used_params[base_key + "_" + str(paramNum)] = param_value  # Uložení parametru a jeho hodnoty                    
                paramNum +=1
            #elif key in ['id', 'inputs', "name"]:  # Vynechání klíčů, které nejsou platné pro vytváření vrstev
             #   continue
            else:
            # Kontrola, zda klíč je v keras_int_params a případný převod na int
                if key in keras_int_params:
                    i[key] = int(value) if isinstance(value, (float, int)) else value
                else:
                    i[key] = value


    return processed_config, used_params


def get_layer(layer, model=None):
    layer = layer.copy()

    keys_to_remove = ['id', 'inputs', "name"]  # Seznam klíčů, které bude třeba odstranit
    # remove random keys and unused ones
    for key in list(layer.keys()):  # Iterace přes kopii klíčů
        if key in keys_to_remove or "Random" in key:
            del layer[key]


    #lp = process_layer_params(layer, paramNum, params)
    lp = layer
    lt = lp["type"].lower()
    lp.pop("type")
    print(lt)
    print("layer params", lp)


    
    layer_switch = {
    'input': Input,
    'dense': Dense,
    'conv2d': Conv2D,
    'generator': Generator,
    'dropout': Dropout,
    'maxpooling2d': MaxPooling2D,
    'lstm': LSTM
    # Můžeš přidat další vrstvy podle potřeby
    }
    
        # Zkontrolujeme, zda je daný typ vrstvy podporován
    if lt in layer_switch:
        layer_class = layer_switch[lt]  # Získáme třídu vrstvy z našeho slovníku

        if lt == "generator": #return instance of generator with required configuration
            print("get_layer pos_layers:", lp["possibleLayers"])
            
            # Vytvoření instance generátoru s odpovídající konfigurací
            gen_instance = layer_class()
            
            #processing udělat až v generátoru
            #processed_generator_layers, used_gen_layer_params = process_parameters(lp["possibleLayers"])
            #print("processed_ged", processed_generator_layers)
            # gen_instance.setRules(lp["possibleLayers"])  # Nastavení pravidel
            gen_instance.setRules(lp["possibleLayers"])  # Nastavení pravidel
            
            #size je kolik vrstev přidáváme, inp je dosud vytvořený model a firstLayer je vrstva z pravidel, která se vezme jako první
            return gen_instance.generateFunc(size=lp["size"], inp=model, firstLayer=lp["firstLayer"])
               
        return layer_class(**lp)  # Vytvoříme instanci vrstvy s parametry
    else:
        raise ValueError(f"Unsupported layer type: {lt}")
        

try:
    trial_code_dir = ''

    # Načtení vrstev a parametrů
    with open(os.path.join(trial_code_dir, 'layers.json'), 'r', encoding="utf-8") as f:
        layers = json.load(f)
        print("layers loaded")


    # Načtení settings
    with open(os.path.join(trial_code_dir,'settings.json'), 'r', encoding="utf-8") as f:
        settings = json.load(f)
        print("settings loaded")

    # Načtení datasetu
    data = np.load(os.path.join(trial_code_dir,'dataset.npz'))
    x_train, x_test = data['x_train'], data['x_test']
    y_train, y_test = data['y_train'], data['y_test']

    # try:
    #     dataset = load_dataset("../datasets/pima-indians-diabetes.csv")
    #     #x = dataset[x_columns] 
    #     x = dataset.iloc[:,0:8] 
    #     y = dataset.iloc[:,8]
    #     #y = dataset[y_columns] 
    #     # print(x)
    #     # print(y)
    #     print("dataset loaded")
    #     x_train, x_test, y_train, y_test = train_test_split(x, y) 
    # except Exception as e:
    #     raise
    # x_train, x_test, y_train, y_test = process_dataset(dataset_path, dataset_config)

    # print("data loaded")
    # print(x_train)
    # print(y_train)

    params = nni.get_next_parameter()
    #print(params)
    print("jsem u načítání parametrů")


    # Vytvoření modelu pomocí existující funkce
    model = create_functional_model(layers, settings, params = params)

    # Načtení dat

    # Trénování modelu
    model.fit(x_train, y_train, epochs=5, verbose=0)

    # Vyhodnocení modelu a reportování výsledků
    #verbose has to be 0, becouse otherwise it doesnt work and you get encoding errors, becouse that makes sense
    val_loss, val_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("val_acc je: ", val_accuracy)
    nni.report_final_result(val_accuracy)
except Exception as e:
    print("Trial exception", e)
    raise