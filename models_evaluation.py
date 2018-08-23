from src.recommender_sys_algorithms import *
from src.evaluation import *
from src.support_functions import *


samples = ['sample_data_1', 'sample_data_2', 'sample_data_3', 'sample_data_4', 'sample_data_5']
sample_data_list = load_samples(samples, data_folder='data')

# Parameters
metrics = ['P@K','R@K','RPrec','AvgP']
similarity_funcs = ['cosine', 'euclidean', 'jaccard', 'pearson']

"""
Format: 
    'model_name_print': {
        'model': 'model_name_in_dictionary',
        'similarity_func': '',
        'parameters': {
            'param1': param_value
        }
    }, 
"""

eval_models ={
    'Item Average':{
        'model': 'ItemAverage',
        'parameters': {}
    },
    'Random':{
        'model': 'Random',
        'parameters': {}
    },
    'Item-Item NN @ Cosine Distance': {
        'model': 'ItemSim',
        'parameters': {
            'sim_method': 'cosine',
            'sim_thresh': 0.26
        }
    },
    'Item-Item NN with Item Attributes': {
        'model': 'ItemSimWithInfo',
        'parameters': {
            'sim_method_r': 'cosine',
            'sim_method_p': 'cosine',
            'alpha': 0.1,
            'sim_thresh': 0.31
        }
    },
    'HSV Embedding with Cosine NN': {
        'model': 'HSVNN',
        'parameters': {
            'sim_method_r': 'cosine',
            'sim_method_p': 'cosine',
            'alpha': 0.01,
            'sim_thresh': 0.31
        }
    },
    'RGB Embedding with Cosine NN': {
        'model': 'RGBNN',
        'parameters': {
            'sim_method_r': 'cosine',
            'sim_method_p': 'cosine',
            'alpha': 0.04,
            'sim_thresh': 0.31
        }
    },
    'Style-embedded with Euclidean NN': {
        'model': 'StyleNN',
        'parameters': {
            'sim_method_r': 'cosine',
            'sim_method_p': 'euclidean',
            'alpha': 0.2,
            'sim_thresh': 0.26
        }
    },
}


def predict(model, data):
    recsys = RecSys(model['model'], data)
    recsys.predict_all(**model['parameters'])
    return recsys.getPrediction()


models = [model for model in eval_models.keys()]
results = []
for model in models:
    print(model)
    for i, data in enumerate(sample_data_list):
        print('Dataset #' + str(i+1))
        start = time.time()
        pred = predict(eval_models[model], data)
        duration = time.time() - start
        result_instance = result(model, i, 'time', duration)  # Save time as one metric
        results.append(result_instance)
        for j, metric in enumerate(metrics):
            evaluation = Evaluation(pred, data.matrix_test, metric)
            measurement = evaluation.evaluate()
            result_instance = result(model, i, metric, measurement)
            results.append(result_instance)
        save_results_in_df(results)
save_evaluation_in_df(results, models, metrics, len(sample_data_list))