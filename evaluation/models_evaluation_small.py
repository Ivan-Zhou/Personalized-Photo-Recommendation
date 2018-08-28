import argparse
from evaluation.src.evaluation import *
from evaluation.src.recommender_sys_algorithms import *
from evaluation.src.support_functions import *

metrics = ['P@K','R@K','RPrec','AvgP']
eval_models = {
    'Item Average': {
        'model': 'ItemAverage',
        'parameters': {}
    },
    'Random': {
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
    'HSV Embedding with Cosine NN': {
        'model': 'HSVNN',
        'parameters': {
            'sim_method_r': 'cosine',
            'sim_method_p': 'cosine',
            'alpha': 0.01,
            'sim_thresh': 0.31
        }
    },
}


def predict(model, data):
    recsys = RecSys(model['model'], data)
    recsys.predict_all(**model['parameters'])
    return recsys.getPrediction()


def evaluate(args):
    data_folder = args['data_folder']
    task = args['task']
    result_folder = args['result_folder']
    job_name = args['job_name']
    assert task=='validation' or task=='test'
    result_path = result_folder + job_name
    sample_data_list = get_data(data_folder, task)

    models = [model for model in eval_models.keys()]
    results = []
    for model in models:
        print(model)
        for i, data in enumerate(sample_data_list):
            print('Dataset #' + str(i + 1))
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
    save_results_in_df(results, folder_path=result_path)
    save_evaluation_in_df(results, models, metrics, len(sample_data_list), folder_path=result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-folder',
        help='Path to get data folder',
        required=True
    )
    parser.add_argument(
        '--task',
        help='For validation or test',
        required=True
    )
    parser.add_argument(
        '--result-folder',
        help='Path to save results',
        default='results'
    )
    parser.add_argument(
        '--job-name',
        help='The name of the job for saving the result',
        default='evaluation'
    )

    args = parser.parse_args()
    evaluate(args.__dict__)
