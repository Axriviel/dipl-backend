from tensorflow import keras

def save_model(model, path): #uložení modelu
    try:
        model.save(path)
        print("Model byl úspěšně uložen")
    except:
        print("Nepodařilo se uložit model")

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