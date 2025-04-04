import joblib

def save_model(model): 
    joblib.dump((model), f'../Model/{model}.pkl')

def load_model(model):
    model_array = joblib.load(f"{model}.pkl")
    return model_array