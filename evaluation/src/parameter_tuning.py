from sklearn.model_selection import ParameterGrid
from evaluation.src.evaluation import *
from evaluation.src.recommender_sys_algorithms import *
from evaluation.src.support_functions import *


results_save_path = 'results/'


class ParameterTuning(object):
    def __init__(self, name, values, description):
        self.name = name  # parameter name
        self.values = values  # a list of values to be tuned
        self.description = description  # description of the parameter


def parameter_tuning(model, metrics, model_parameters, target_parameter, data_list):
    if not isinstance(data_list, list):  # one individual data entity is passed in
        data_list = [data_list]

    print('Model: ' + model + ', parameter for tuning: ' + target_parameter.description)
    identification = model + '_' + target_parameter.name

    def get_pred_on_data_list(model, data_list, params):
        prediction_list = []
        for data in data_list:
            recsys = RecSys(model, data)
            recsys.predict_all(**params)
            pred = recsys.getPrediction()
            prediction_list.append(pred)
        return prediction_list

    def get_evaluation_result(metric, prediction_list, data_list):
        """
        Prediction in prediction_list is matched with the corresponding data in the data_list on the same index
        """
        results = np.zeros(len(prediction_list))
        for i, pred in enumerate(prediction_list):
            evaluation = Evaluation(pred, data_list[i].matrix_test, metric, k=10)
            results[i] = evaluation.evaluate()
        metric_goal = evaluation._getMetricGoal(metric)
        return np.mean(results), results, metric_goal  # Get the average result from all the predictions

    def update_best_results(best_results, metric, result, parameter_set):
        best_results[metric]['best_result'] = result
        best_results[metric]['best_parameter_set'] = parameter_set
        return best_results

    metrics_list = []
    parameters_list = []
    best_results = {}
    for metric in metrics:
        best_results[metric] = {
            'best_result': 0,
            'best_parameter_set': {}
        }

    metric_goal_dict = {}  # list to record the goal of each metric (max or min)

    param_grid = {}
    for param in model_parameters.keys():
        param_grid[param] = [model_parameters[param]] if not isinstance(model_parameters[param], list) else model_parameters[param]
    param_grid[target_parameter.name] = target_parameter.values
    grid = ParameterGrid(param_grid)

    results_records = []
    mean_results = np.zeros((len(metrics), len(target_parameter.values)))
    i = 0
    for params in grid:
        prediction_list = get_pred_on_data_list(model, data_list, params)
        for j, metric in enumerate(metrics):
            mean_result, results, metric_goal = get_evaluation_result(metric, prediction_list, data_list)
            metric_goal_dict[metric] = metric_goal
            metrics_list.append(str(i) + '-' + metric)
            results_records.append(results)
            parameters_list.append(params)
            mean_results[j, i] = mean_result
            if metric_goal == 'max':
                if best_results[metric]['best_result'] < mean_result:
                    best_results = update_best_results(best_results, metric, mean_result, params)
            else:  # metric's goal == 'min'
                if best_results[metric]['best_result'] > mean_result:
                    best_results = update_best_results(best_results, metric, mean_result, params)
        i += 1
        # Save results dataframe
        df_save_path = results_save_path + identification + '.csv'
        results_df = pd.DataFrame(results_records, columns=['dataset - ' + str(i) for i in range(len(data_list))])
        results_df['Metrics'] = metrics_list
        results_df['Parameter Value'] = parameters_list
        results_df.to_csv(df_save_path)

    # Save plot
    img_save_path = results_save_path + identification + '.png'
    plot_multiple_lines(target_parameter.values, mean_results, metrics, target_parameter.description,
                        'Metrics', 'Parameter Tuning', img_save_path)



    # Save best parameter evaluation
    text_file_path = results_save_path + identification + '.txt'
    file = open(text_file_path, "w")

    for metric in metrics:
        file.write('For metric ' + metric + ', the best value is ' + str(best_results[metric]['best_result']) + '\n')
        file.write('which is achived at: \n')
        file.write(target_parameter.name + ': ' + str(best_results[metric]['best_parameter_set'][target_parameter.name]) + '\n')
        file.write('\n')
    file.close()


def parameters_tuning(model, metrics, target_parameters, data_list):
    print('Model: ' + model)
    identification = model + '_parameters'

    metrics_list = []
    parameters_list = []
    results_records = []

    best_results = {}
    for metric in metrics:
        best_results[metric] = {
            'best_result': 0,
            'best_parameter_set': {}
        }

    if not isinstance(data_list, list):  # one individual data entity is passed in2
        data_list = [data_list]

    def update_best_results(best_results, metric, result, parameter_set):
        best_results[metric]['best_result'] = result
        best_results[metric]['best_parameter_set'] = parameter_set
        return best_results

    def get_pred_on_data_list(model, data_list, params):
        prediction_list = []
        for data in data_list:
            recsys = RecSys(model, data)
            recsys.predict_all(**params)
            pred = recsys.getPrediction()
            prediction_list.append(pred)
        return prediction_list

    def get_evaluation_result(metric, prediction_list, data_list):
        """
        Prediction in prediction_list is matched with the corresponding data in the data_list on the same index
        """
        results = np.zeros(len(prediction_list))
        for i, pred in enumerate(prediction_list):
            evaluation = Evaluation(pred, data_list[i].matrix_test, metric, k=10)
            results[i] = evaluation.evaluate()
        metric_goal = evaluation._getMetricGoal(metric)
        return np.mean(results), results, metric_goal  # Get the average result from all the predictions


    param_grid = {}
    count = 0
    for target_parameter in target_parameters:
        param_grid[target_parameter.name]=target_parameter.values
    grid = ParameterGrid(param_grid)
    for params in grid:
        count += 1
        prediction_list = get_pred_on_data_list(model, data_list, params)
        for j, metric in enumerate(metrics):
            result, results, metric_goal = get_evaluation_result(metric, prediction_list, data_list)
            results_records.append(results)
            metrics_list.append(str(count) + '-' + metric)
            parameters_list.append(params)

            if metric_goal == 'max':
                if best_results[metric]['best_result'] < result:
                    best_results = update_best_results(best_results, metric, result, params)
            else:  # metric's goal == 'min'
                if best_results[metric]['best_result'] > result:
                    best_results = update_best_results(best_results, metric, result, params)

        # Save results dataframe
        df_save_path = results_save_path + identification + '.csv'
        results_df = pd.DataFrame(results_records, columns=['dataset - ' + str(i) for i in range(len(data_list))])
        results_df['Metrics'] = metrics_list
        results_df['Parameter Value'] = parameters_list
        results_df.to_csv(df_save_path)

    # Save best parameter evaluation
    text_file_path = results_save_path + identification + '.txt'
    file = open(text_file_path, "w")
    for metric in metrics:
        file.write('For metric ' + metric + ', the best value is ' + str(best_results[metric]['best_result']) + '\n')
        file.write('which is achived at: \n')
        for param in target_parameters:
            file.write(param.name + ': ' + str(best_results[metric]['best_parameter_set'][param.name]) + '\n')
        file.write('\n')
    file.close()
    for target_parameter in target_parameters:
        if len(target_parameter.values) > 1:  # Filter out the parameters not for tuning
            parameter_tuning(model, metrics, best_results['P@K']['best_parameter_set'],
                             target_parameter, data_list)
