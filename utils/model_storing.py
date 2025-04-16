from tensorflow import keras
import os
#save model
def save_model(model, user_id, model_id): 
    try:
        # path to user folder
        user_folder = os.path.join("userModels", str(user_id))
        
        # create if not found
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        
        # save model to folder
        model_path = os.path.join(user_folder, f"model_{model_id}.keras")
        model.save(model_path)
        
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Nepodařilo se uložit model: {str(e)}")

#load the model
def load_model(path):
    try:
        model = keras.models.load_model(path)
        print("Model byl úspěšně načten")
        return model
    except Exception as e:
        print(e+ "Nepodařilo se načíst model")
            