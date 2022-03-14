from util import ProcessPool
import pandas as pd

def predict_fn(n, p, user_id, item_id, decimals): 
    return (n, p.predict(user_id, item_id, decimals))


def predict(predictors, user_id, item_id, n_processes=None, decimals=2):
    params  = [(p.name, p, user_id, item_id, decimals) for p in predictors]

    return pd.DataFrame(
        data=ProcessPool(n_processes).run(predict_fn, params),
        columns=['Predictor', 'Prediction']
    ).set_index('Predictor')
