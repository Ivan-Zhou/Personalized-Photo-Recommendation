import argparse
from data.load_data import *
from evaluation.src.evaluation import *
from evaluation.src.recommender_sys_algorithms import *
from evaluation.src.support_functions import *


metrics = ['P@K','R@K','RPrec','AvgP']


def predict(model, data, *arg):
    recsys = RecSys(model, data)
    recsys.predict_all(arg)
    return recsys.getPrediction()


def evaluate(args):
    data_folder = args['data_folder']
    task = args['task']
    model_name = args['model_name']
    result_folder = args['result_folder']
    job_name = args['job_name']
    assert task=='validation' or task=='test'
    result_path = result_folder + job_name
    sample_data_list = get_data(data_folder, task)
    results = []
    for i, data in enumerate(sample_data_list):
        start = time.time()
        pred = predict(model_name, data)
        duration = time.time() - start
        result_instance = result(model_name, i+1, 'time', duration)  # Save time as one metric
        result_instance.print_result()
        results.append(result_instance)
        for j, metric in enumerate(metrics):
            evaluation = Evaluation(pred, data.matrix_test, metric)
            measurement = evaluation.evaluate()
            result_instance = result(model_name, i+1, metric, measurement)
            result_instance.print_result()
            results.append(result_instance)
    save_results_in_df(results, df_name='evaluate_' + model_name, folder_path=result_path)


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
        '--model-name',
        help='The name of the model',
        required=True
    )
    parser.add_argument(
        '--sim-method-r',
        help='The Similarity Metric for User-Item Interaction',
        default='cosine'
    )
    parser.add_argument(
        '--sim-method-p',
        help='The similarity metric for attributes',
        default='cosine',
    )
    parser.add_argument(
        '--alpha',
        help='The weight on attributes',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--sim_threshold',
        help='threshold on similarity',
        type=float,
        default=0.1
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
