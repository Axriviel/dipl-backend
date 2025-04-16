from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D, Dropout, MaxPooling2D, LSTM, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from utils.task_protocol_manager import task_protocol_manager
from tensorflow import keras
from flask import session
import time
import random
from utils.dataset_storing import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils.task_progress_manager import growth_limiter_manager

#creates optimized model based on selected algorithm
#returns  best model, best metric value, best metric history, best used parameters
def create_optimized_model(layers, settings, dataset_path, dataset_config, opt_data={}):
    try:
        user_id = session["user_id"]
        num_runs = settings.get("k-fold", 1)

        task_protocol_manager.log_item(user_id, "limit_growth", settings.get("limit_growth"))
        opt_method = settings["opt_algorithm"]
        max_progress = 1
        if opt_method == "random":
            max_progress = settings.get("max_models", 5)
        elif opt_method == "genetic":
            max_progress = settings.get("GA", {}).get("generations", 10)
        else:
            max_progress = settings.get("max_models", 5)

        growth_limiter_manager.set_growth(user_id, settings.get("limit_growth", "none"), max_progress)


        task_type = opt_data.get("task_type", "")

        input_shape, x_train, x_test, y_train, y_test = process_dataset(dataset_path, dataset_config, settings, task_type)
        output_shape = y_train.shape

        #set last layer parameters based on task and dataset (for automated designer)
        if task_type == "binary classification":
            pass
            # print("old y_traiin", y_train)
            # y_train = y_train[:1]
            # print("new y_train", y_train)
            # y_test = y_test[:1]
        elif task_type == "multiclass classification" or task_type == "image multiclass classification":
            layers[-1]["units"] = output_shape[1]
           
        # set input shape into input layer
        layers[0]["shape"] = input_shape

        if opt_method == "random":
            try:
                from optimizers.random_optimizer import random_search

                b_model, b_metric_val, b_metric_history, used_params = random_search(layers, settings, 
                                                          x_train=x_train, y_train=y_train, 
                                                          x_val=x_test, y_val=y_test, 
                                                          num_models=settings["max_models"], num_runs=num_runs, 
                                                          threshold=settings["es_threshold"], 
                                                          monitor_metric=settings["monitor_metric"])
                print(f"Best model found with {settings['monitor_metric']} : {b_metric_val}")
                return b_model, b_metric_val, b_metric_history, used_params

            except Exception as e:
                print("GA essentials exception: ", e)
                raise


        elif opt_method == "genetic":
            try:
                from optimizers.genetic_optimizer import genetic_optimization
                ga_config = settings["GA"]

                b_model, b_metric_val, b_metric_history, used_params = genetic_optimization(
                    layers, 
                    settings, 
                    x_train=x_train, 
                    y_train=y_train, 
                    x_val=x_test, 
                    y_val=y_test, 
                    num_runs = num_runs,
                    population_size=ga_config["populationSize"], 
                    num_generations=ga_config["generations"], 
                    num_parents=ga_config["numParents"], 
                    mutation_rate=ga_config["mutationRate"], 
                    selection_method=ga_config["selectionMethod"],
                    threshold=settings["es_threshold"],
                    monitor_metric=settings["monitor_metric"],
                    user_id=user_id
                )

                return b_model, b_metric_val, b_metric_history, used_params
            except Exception as e:
                print("GA essentials exception: ", e)
                raise
        

        elif opt_method == "bayesian":
            try:
                from optimizers.bayes_optimizer import bayesian_optimization

                b_model, b_metric_val, b_metric_history, used_params = bayesian_optimization(
                    layers,
                    settings,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_test,
                    y_val=y_test,
                    max_trials=settings.get("max_models", 15),
                    num_runs=num_runs,
                    threshold=settings["es_threshold"],
                    monitor_metric=settings["monitor_metric"]
                )

                return b_model, b_metric_val, b_metric_history, used_params


            except Exception as e:
                raise

        elif opt_method == "nni":
            try:
                from nni.experiment import Experiment
                import os
                import json

                nni_config = settings["NNI"]

                search_space = generate_nni_search_space(layers)
                # print("search space:")
                # print(search_space)

                trial_code_dir = 'nni_trials'
                os.makedirs(trial_code_dir, exist_ok=True)
                with open(os.path.join(trial_code_dir, 'layers.json'), 'w', encoding="utf-8") as f:
                    json.dump(layers, f)

                with open(os.path.join(trial_code_dir, 'settings.json'), 'w', encoding="utf-8") as f:
                    json.dump(settings, f)

                import numpy as np
                np.savez(os.path.join(trial_code_dir, 'dataset.npz'), 
                         x_train=x_train, x_test=x_test, 
                         y_train=y_train, y_test=y_test)
                # print("vse ulozeno")

                experiment = Experiment('local')
                experiment.config.search_space = search_space
                experiment.config.trial_command = 'python trial.py'
                experiment.config.trial_code_directory = trial_code_dir
                experiment.config.trial_concurrency = nni_config["nni_concurrency"]
                experiment.config.max_trial_number = nni_config["nni_max_trials"]
                experiment.config.tuner.name = nni_config["nni_tuner"]

                print("Starting NNI experiment...")
                experiment.run(port=8080)

                while True:
                    status = experiment.get_status()
                    print(status)
                    if status == 'DONE':
                        print("Experiment completed successfully!")
                        break
                    elif status == 'STOPPED':
                        print("Experiment was stopped manually.")
                        break
                    elif status == 'ERROR':
                        print("Experiment encountered an error.")
                        break
                    time.sleep(5)  # wait 5 seconds before checking again

                #get experiment results
                exp_data = experiment.export_data()

                #find best result
                best_trial = get_best_trial(exp_data, optimize_mode="maximize")

                best_params = best_trial.parameter
                b_metric_val = best_trial.value
                b_model, used_params = create_functional_model(layers, settings, params = [best_params, {}, []])
                b_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

            except Exception as e:
                raise
            finally:
                experiment.stop()
                pass

            #not supported yet
            b_metric_history = []

            return b_model, b_metric_val, b_metric_history, used_params

        elif opt_method == "tagging":
            try:
                from optimizers.tagging_optimizer import tagging_optimization
                b_model, b_metric_val, b_metric_history, used_params = tagging_optimization(
                    layers, 
                    settings, 
                    x_train=x_train, 
                    y_train=y_train, 
                    x_test=x_test, 
                    y_test=y_test,
                    num_of_models=5, 
                    num_runs=num_runs, 
                    threshold=settings["es_threshold"], 
                    opt_data=opt_data
                )
                pass
            except Exception as e:
                raise

            return b_model, b_metric_val, b_metric_history, used_params
    except Exception as e:
        raise e

#used for nni to find best trial in the list (since there is no method for that)
def get_best_trial(exp_data, optimize_mode="maximize"):
    best_trial = None
    best_value = float('-inf') if optimize_mode == "maximize" else float('inf')

    for trial in exp_data:
        value = trial.value  
        if value is not None:
            if (optimize_mode == "maximize" and value > best_value) or \
               (optimize_mode == "minimize" and value < best_value):
                best_value = value
                best_trial = trial

    if best_trial:
        return best_trial
    else:
        raise ValueError("Žádný validní trial nebyl nalezen.")


#used to create search space for NNI based on layer config received
def generate_nni_search_space(layers):
    """Generuje vyhledávací prostor pro NNI na základě vrstev."""
    try:
        search_space = {}
    
        for layer in layers:
            for key, value in layer.items():
                if 'Random' in key:
                    base_key = key.replace('Random', '')
                    if value['type'] == 'numeric':
                        search_space[f'{base_key}_{layer["id"]}'] = {
                            '_type': 'quniform',
                            '_value': [value['min'], value['max'], value['step']]
                        }
                    elif value['type'] == 'text':
                        search_space[f'{base_key}_{layer["id"]}'] = {
                            '_type': 'choice',
                            '_value': value['options']
                        }
        return search_space
    except Exception as e:
        print("generate_nni_search_space exception ", e)


def process_layer_params_nni(layer, params):
    try:
        processed_params = {}

        for key, value in layer.items():
            if 'Random' in key:
                base_key = key.replace('Random', '')  
                processed_params[base_key] = params.get(f'{base_key}_{layer["id"]}', value.get('default'))
            elif key in ['id', 'inputs', 'name']:  
                continue
            else:
                processed_params[key] = value

        return processed_params
    except Exception as e:
        raise


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def process_dataset(dataset_path, dataset_config, model_settings, task_type=""):
    try:
        dataset = load_dataset(dataset_path)

        if dataset_path.endswith('.npz'):
            x_train, y_train = dataset["x_train"], dataset["y_train"]
            input_shape = list(x_train.shape[1:])
            # output_shape = list(y_train.shape[1:])
            x_test, y_test = dataset.get("x_test", None), dataset.get("y_test", None)
            return input_shape, x_train, x_test, y_train, y_test

        elif dataset_path.endswith('.csv'):
            dataset = convert_numeric_columns(dataset)

            if dataset_config.get("x_columns"):
                x = dataset[dataset_config["x_columns"]]
                input_shape = [len(dataset_config["x_columns"])]
            else:
                x = dataset.iloc[:, :dataset_config["x_num"]]
                input_shape = [dataset_config["x_num"]]

            if dataset_config.get("y_columns"):
                y = dataset[dataset_config["y_columns"]]
            else:
                y = dataset.iloc[:, dataset_config["y_num"] - 1]

            # convert series to pandas
            if isinstance(y, pd.Series):
                y = y.to_frame()

            x_onehot_cols = dataset_config.get("one_hot_x_columns", [])
            y_onehot_cols = dataset_config.get("one_hot_y_columns", [])

            categorical_y = []

            # ======== ENCODING Y =========

            # if we have task of "binary_classification, we do not need both columns"
            # unless doing some specification for which layer is output (would require user input), there is not way to make similar check for semi
            drop_first = (task_type == "binary_classification")
            if dataset_config.get("encode_y", False) and model_settings.get("loss") != "sparse_categorical_crossentropy":
                print("Full y encoding (encode_y = True)")
                y, _ = encode_labels(y.iloc[:, 0]) 
                categorical_y = [y.columns[0]] if isinstance(y, pd.DataFrame) else []
            elif y_onehot_cols:
                print("Encoding specific y columns:", y_onehot_cols)
                y_encoded = y.copy()
                for col in y_onehot_cols:
                    encoded = apply_one_hot_encoding(y[[col]], [col], drop_first=drop_first)
                    y_encoded = y_encoded.drop(columns=col).join(encoded)
                y = y_encoded
                categorical_y = y_onehot_cols
            else:
                print("Auto-detecting text columns in y")
                categorical_y = detect_text_columns(y)
                if categorical_y:
                    y = apply_one_hot_encoding(y, categorical_y, drop_first=drop_first)

            # ========== ENCODING X ==========
            if x_onehot_cols:
                x = apply_one_hot_encoding(x, x_onehot_cols)

            categorical_x = detect_text_columns(x)
            x = apply_one_hot_encoding(x, categorical_x, True)

            input_shape = [x.shape[1]]

            print("encoded y", y)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=dataset_config["test_size"])

            user_id = session.get("user_id")

            columns_info = {
                "x_columns": dataset_config.get("x_columns") or list(dataset.columns[:dataset_config["x_num"]]),
                "y_columns": dataset_config.get("y_columns") or [dataset.columns[dataset_config["y_num"] - 1]],
                "one_hot_encoded_x": categorical_x,
                "one_hot_encoded_y": categorical_y
            }
            task_protocol_manager.log_dict(user_id, columns_info)

            return input_shape, x_train, x_test, y_train, y_test

    except Exception as e:
        print("Error processing dataset:", e)
        raise


def convert_numeric_columns(df):
    """ Převádí sloupce, které obsahují pouze čísla, na float/int. """
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass
    return df


def detect_text_columns(df):
    """ Detekuje sloupce, které obsahují text (object dtype). """
    return df.select_dtypes(include=['object']).columns.tolist()


def detect_low_unique_columns(df, threshold=10):
    """ Detekuje číselné sloupce, které mají méně než `threshold` unikátních hodnot. """
    categorical_columns = []
    for col in df.select_dtypes(include=['int', 'float']).columns:
        if df[col].nunique() <= threshold:
            categorical_columns.append(col)
    return categorical_columns


def apply_one_hot_encoding(df, categorical_columns, drop_first = False):
    """ Aplikuje OneHotEncoder pouze na zadané sloupce. """
    if not categorical_columns:
        return df

    encoder = OneHotEncoder(drop="first" if drop_first else None, sparse_output=False, handle_unknown="ignore")
    encoded_cols = encoder.fit_transform(df[categorical_columns])

    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_columns), index=df.index)

    df = df.drop(columns=categorical_columns)
    df = pd.concat([df, encoded_df], axis=1)

    return df

def encode_labels(y):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(-1, 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    class_labels = label_encoder.classes_
    columns = [f"{y.name}_{cls}" for cls in class_labels]

    return pd.DataFrame(onehot_encoded, columns=columns), label_encoder


def generate_random_value(random_info):
    try:
        """Generate a random value based on the provided randomization info."""
        if random_info['type'] == 'numeric-test':
            return random.randint(random_info['min'], random_info['max'])
        elif random_info['type'] == 'numeric':
            min_val = random_info['min']
            max_val = random_info['max']
            step = random_info.get('step', 1)
    
            is_float = isinstance(step, float) or isinstance(min_val, float) or isinstance(max_val, float)
    
            count = int((max_val - min_val) / step) + 1
            values = [round(min_val + i * step, 10) for i in range(count)]
    
            return random.choice(values if is_float else list(map(int, values)))
        elif random_info['type'] == 'text':
            return random.choice(random_info['options'])
        return None
    except:
        raise ValueError("Could not generate random")

def use_limit_growth_function(random_info, used_growth_function, generation, max_generations):
    # cant limit text
    if random_info['type'] not in ['numeric', 'numeric-test']:
        return random_info

    # get values
    original_min = random_info.get('original_min', random_info['min'])
    original_max = random_info.get('original_max', random_info['max'])
    step = random_info.get('step', 1)

    # save original values too
    random_info['original_min'] = original_min
    random_info['original_max'] = original_max

    # count progress to maximum
    progress = generation / max_generations if max_generations > 0 else 1.0

    if used_growth_function == "linear":
        new_max = original_min + progress * (original_max - original_min)

    elif used_growth_function == "square":
        new_max = original_min + (original_max - original_min) * (progress ** 2)

    elif used_growth_function == "log":
        import math
        scale = (original_max - original_min) / math.log(max_generations + 1)
        new_max = original_min + scale * math.log(generation + 1)
    else:
        new_max = original_max  # fallback

    # make sure the value is at least one step possible
    min_allowed_max = original_min + step
    if new_max < min_allowed_max:
        new_max = min_allowed_max
        
    # round down
    if all(isinstance(x, int) for x in [original_min, original_max, step]):
        new_max = int(new_max)

    # update random_info
    random_info['min'] = original_min
    random_info['max'] = min(new_max, original_max)

    return random_info


#process and return a list of parameters
def process_parameters(config, params=None, keras_int_params=None):
    import copy
    processed_config = copy.deepcopy(config)
    used_params = {}  #save used params
    paramNum = 0
    user_id = session.get("user_id")

    if keras_int_params is None:
        from config.settings import Config 
        keras_int_params = Config.KERAS_INT_PARAMS
    
    for i in processed_config:
        for key, value in i.items():
        # replace random keys with value
            if 'Random' in key:
                base_key = key.replace('Random', '') 
            
                if params and paramNum < len(params):
                    param_value = list(params.values())[paramNum]

                # convert to int when necessary, becouse quniform returns float every time
                    if base_key in keras_int_params:
                        param_value = int(param_value)
                    i[base_key] = param_value
                else:
                    # value contains configuration for keys with Random

                    growth_limited_value = use_limit_growth_function(value, growth_limiter_manager.get_growth_function(user_id), 
                                                                     growth_limiter_manager.get_current_progress(user_id),
                                                                     growth_limiter_manager.get_max_progress(user_id))
                    # log limits used in current epoch
                    epoch = task_protocol_manager.get_log(user_id).get_or_create_epoch(epoch_number = growth_limiter_manager.get_current_progress(user_id))
                    epoch.limits.append(growth_limited_value)

                    param_value = generate_random_value(growth_limited_value)
                    i[base_key] = param_value
                    
                used_params[base_key + "_" + str(paramNum)] = param_value   
                paramNum +=1
            else:
            # convert to int if necessary
                if key in keras_int_params:
                    i[key] = int(value) if isinstance(value, (float, int)) else value
                else:
                    i[key] = value


    return processed_config, used_params
    
def get_layer(layer, model=None, optional_param=None):
    layer = layer.copy()

    #remove unnecessary keys
    keys_to_remove = ['id', 'inputs', "name"]  
    # remove random keys and unused ones
    for key in list(layer.keys()):
        if key in keys_to_remove or "Random" in key:
            del layer[key]

    lp = layer
    lt = lp["type"].lower()
    lp.pop("type")

    layer_switch = {
    'input': Input,
    'dense': Dense,
    'conv2d': Conv2D,
    'generator': Generator,
    'dropout': Dropout,
    'maxpooling2d': MaxPooling2D,
    'lstm': LSTM,
    "flatten": Flatten,
    "batchnormalization": BatchNormalization
    # add more layers if supported
    }
    
    if lt in layer_switch:
        layer_class = layer_switch[lt] 

        if lt == "generator": #return instance of generator with required configuration
            
            # create generator instance with config
            gen_instance = layer_class()
            gen_instance.setRules(lp["possibleLayers"])  

            if optional_param is not None:
                strct = gen_instance.generateFunc(size=lp["size"], inp=model, firstLayer=lp["firstLayer"], config=optional_param)
                parm = gen_instance.used_struct
                return [strct, parm]
            
            else:
                #set generator rules
                gen_instance.setRules(lp["possibleLayers"])
            
                #size - how many layers possible, inp - model, firstLayer - first layer from rules
                strct = gen_instance.generateFunc(size=lp["size"], inp=model, firstLayer=lp["firstLayer"], config=optional_param)
                parm = gen_instance.used_struct
                return [strct, parm]
            
        return layer_class(**lp)  #return layer
    else:
        raise ValueError(f"Unsupported layer type: {lt}")
        


#used to create the model itself
def create_functional_model(layers, settings, params = None):
    try:
        #sending as list so that the change is written here
        layer_outputs = {}

        #used only for generator - params if creating new and number to know which one it is for regenerating from config
        generator_settings = [None]
        used_generator_params = []
        generator_number = 0

        #process parameters of layers and models settings
        if params == None:
            #proces layers
            processed_layers, used_layers_params = process_parameters(layers)
            #process parameters
            processed_settings, used_settings_params = process_parameters([settings])
            #remove outer list
            processed_settings = processed_settings[0]
        else:
            processed_layers, used_layers_params = process_parameters(layers, params = params[0])
            processed_settings, used_settings_params = process_parameters([settings], params = params[1])
            processed_settings = processed_settings[0]
            generator_settings = params[2]

        unresolved_layers = processed_layers.copy() 

        while unresolved_layers:
            for layer in unresolved_layers[:]: 

                # create input layer
                if layer["type"] == "Input" and layer["id"] not in layer_outputs:
                    input_layer = get_layer(layer, model=None)
                    layer_outputs[layer['id']] = input_layer
                    unresolved_layers.remove(layer)
                    continue
                # make sure all inputs for layer are created
                if all(input_id in layer_outputs for input_id in layer['inputs']):
                    input_tensors = [layer_outputs[input_id] for input_id in layer['inputs']]

                    # concatenate multiple inputs if needed
                    input_tensor = input_tensors[0] if len(input_tensors) == 1 else Concatenate()(input_tensors)

                    if layer["type"] == "Generator":
                        #generator return whole model, so we need to treat it as such
                        #return model at [0] and replicable process at [1]
                        if len(generator_settings)<generator_number+1:
                            gen_output = get_layer(layer, input_tensor)
                        else:
                            gen_output = get_layer(layer, input_tensor, optional_param=generator_settings[generator_number])

                        generator_number += 1
                        gen_used_config = gen_output[1]
                        #save the particular generator info into list in order
                        used_generator_params.append(gen_used_config)


                        output_tensor = gen_output[0]
                    else:
                        new_layer = get_layer(layer, input_tensor)
                        output_tensor = new_layer(input_tensor)

                    # save output
                    layer_outputs[layer['id']] = output_tensor
                    unresolved_layers.remove(layer)  #remove layer from unresolved

        # specification of the model
        input_layer = layer_outputs[processed_layers[0]['id']]
        output_tensor = list(layer_outputs.values())[-1] #last layer is the output one

        model = Model(inputs=input_layer, outputs=output_tensor)
        # compile model
        model.compile(
            optimizer=processed_settings['optimizer'], 
            loss=processed_settings['loss'], 
            metrics=processed_settings['metrics']
        )

        return model, [used_layers_params, used_settings_params, used_generator_params]
    except Exception as e:
        raise e


def remove_outer_list(x):
    return x[0]

class Generator:
    def __init__(self, attempts=5):
        self.rules = {}
        self.attempts = attempts
        #to store used structure
        self.used_struct = {}

    def setRules(self, layers):
        self.layers = layers
        rules = {}
        for layer in layers:
            layer_id = layer["id"]

            if len(layers) == 1:
                layer["inputs"] = layer["id"]

            possible_next_layers = [
                (l["type"], l["id"]) for l in layers if l["id"] in layer.get("inputs", [])
            ]

            rules[layer_id] = {
                "layer": layer,
                "next_layers": possible_next_layers
            }
        self.rules = rules
        

    def generateFunc(self, size, inp, out=None, firstLayer=None, config=None):
        inpStruct = inp
        attempts = self.attempts

        used_parameters = []
        used_layers_sequence = []

        while attempts > 0:
            try:
                struct = inpStruct
                
                if config is not None and len(config)>0:
                    use_rules = config["used_rules"]
                    use_sequence = config["layers_sequence"]
                    use_params = config["used_parameters"]
                    
                    for i in range(len(use_sequence)):
                        rule = use_rules[use_sequence[i]]
                        processed_rule_layer, _ = process_parameters([rule["layer"]], use_params[i])
                        
                        used_parameters.append(use_params[i])
                        used_layers_sequence.append(use_sequence[i])
                        
                        new_layer = get_layer(remove_outer_list(processed_rule_layer), struct)
                        struct = new_layer(struct) if callable(new_layer) else new_layer
                else:
                    current_layer_id = firstLayer if firstLayer else list(self.rules.keys())[0]
                
                    for i in range(size):
                        rule = self.rules.get(current_layer_id, None)
                        if rule is None:
                            raise ValueError(f"Layer with ID {current_layer_id} not found in rules.")
                    
                        processed_rule_layer, used_rule_param = process_parameters([rule["layer"]])
                        used_parameters.append(used_rule_param)
                        #add
                        used_layers_sequence.append(current_layer_id)

                        # create layer with processed params
                        new_layer = get_layer(remove_outer_list(processed_rule_layer), struct)
                        struct = new_layer(struct) if callable(new_layer) else new_layer

                        # pick next layer based on possible follows
                        if rule["next_layers"]:
                            next_layer_id = random.choice([layer_id for _, layer_id in rule["next_layers"]])
                            current_layer_id = next_layer_id
                        else:
                            break

                
                if out is not None:
                    self.used_struct["used_parameters"] = used_parameters
                    self.used_struct["layers_sequence"] = used_layers_sequence
                    self.used_struct["used_rules"] = self.rules
                    self.used_struct["used_layers"] = self.layers
                    
                    return out(struct)
                else:
                    self.used_struct["used_parameters"] = used_parameters
                    self.used_struct["layers_sequence"] = used_layers_sequence
                    self.used_struct["used_rules"] = self.rules
                    self.used_struct["used_layers"] = self.layers
                    
                    return struct

            except Exception as e:
                print(f"Error during generation: {e}")
                attempts -= 1

        print("Could not generate working structure within the limit.")
        return inpStruct

