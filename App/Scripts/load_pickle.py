import pickle

def load_pickle():
    model = pickle.load(open('https://raw.githubusercontent.com/bumpansy/Heart-Failure-Prediction/blob/830f397598e8b0cb31bb41c9b3db58568e5b2cad/Model/new_model.pkl', 'rb'))
    return model

load_pickle()
print('ran success')