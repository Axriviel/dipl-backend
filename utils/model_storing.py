from tensorflow import keras
import os

def save_model(model, user_id, model_id):  # Přidáme model_id jako argument
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


def load_model(path): #načtení modelu
    try:
        model = keras.models.load_model(path)
        print("Model byl úspěšně načten")
        return model
    except Exception as e:
        print(e+ "Nepodařilo se načíst model")
            
# def compareModels(m1, m2, metric, hilo):#model, model, index_metriky, high/low
#     x_test = settings.x_test
#     y_test = settings.y_test
    
#     if x_test or y_test is None:
#         print("Nevhodně vyplněno nastavení x_test a y_test")
#         pass
    
#     val1 = m1.evaluate(x_test, y_test, verbose=0)
#     m1Val = val1[metric]
#     val2 = m2.evaluate(x_test, y_test, verbose=0)
#     m2Val = val2[metric]  
    
#     if m1Val == m2Val:
#         print("Oba modely dosahly stejnych vysledku")
#     elif m1Val > m2Val:
#         print("První model dosáhl lepšího výsledku")
#     elif m1Val < m2Val:
#         print("Druhý model dosáhl lepšího výsledku")
#     else:
#         print("Došlo k neočekávanému výsledku")