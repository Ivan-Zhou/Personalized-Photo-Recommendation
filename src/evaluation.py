from src.support_functions import *

class Evaluation(object):
    def __init__(self, prediction, testSet, metric, k=10):
        """
            INPUT:
                metric: string. from['RMSE','P@K','R@K']
        """
        self.metric_name = metric
        self.metric = self._getMetric(self.metric_name)
        self.prediction = prediction.todense()
        self.testSet = testSet.todense()
        self.k = k

    def _getMetric(self, metric_name):
        switcher = {
            'P@K': self.patk,
            'R@K': self.ratk,
            'RPrec': self.rprec,
            'AvgP': self.avgp,
        }
        return switcher[metric_name]

    def _getMetricGoal(self, metric_name):
        metricGoalDict = {
            'P@K': 'max',
            'R@K': 'max',
            'RPrec': 'max',
            'AvgP': 'max',
        }
        return metricGoalDict[metric_name]

    def patk(self, prediction, testSet):
        """
        Precision at k: # true positive @ k / # recommended @ k
        """
        k = self.k
        ranks = np.argsort(prediction, axis=1)  # argsort applied on each row; ascending order
        ranks_k = ranks[:, -k:]  # Get the top k
        test_k = testSet[np.arange(np.shape(testSet)[0])[:, np.newaxis], ranks_k]
        pre_k = np.count_nonzero(test_k) / test_k.size
        return pre_k

    def ratk(self, prediction, testSet):
        """
        recall at k: # true positive @k / # relevant in total
        :param prediction: prediction matrix
        :param testSet: testset in matrix
        :return: recall @ k
        """
        k = self.k
        ranks = np.argsort(prediction, axis=1)  # argsort applied on each row; ascending order
        ranks_k = ranks[:, -k:]  # Get the top k
        test_k = testSet[np.arange(np.shape(testSet)[0])[:, np.newaxis], ranks_k]
        rec_k = np.count_nonzero(test_k) / np.count_nonzero(testSet)
        return rec_k

    def rprec(self, prediction, testSet):
        """
            R-Precision
        """
        # Initialize sum and count vars for average calculation
        RPrec_list = []
        ranks = np.argsort(prediction, axis=1)  # argsort applied on each row; ascending order
        for userID in range(len(prediction)):
            nRelevant = np.count_nonzero(testSet[userID, :])  # Get the # relevant, used as k
            # Ignore user if has no ratings in the test set
            if (nRelevant == 0):
                continue
            test_k = testSet[userID, ranks[userID, -nRelevant:]]  # Get labels for the recommended items
            RPrec = np.count_nonzero(test_k) / nRelevant
            # Update sum and count
            RPrec_list.append(RPrec)
        # Return average P@k
        return np.mean(RPrec_list)

    def avgp(self, prediction, testSet):
        """
        Average Precision =  sum_k {(P(k) * rel(k))}/ n_R
        """
        AvgPrec_list = []
        ranks = np.argsort(prediction, axis=1)  # argsort applied on each row; ascending order
        for userID in range(len(prediction)):
            nRelevant = np.count_nonzero(testSet[userID, :])  # Get the # relevant, used as k
            # Ignore user if has no ratings in the test set
            if (nRelevant == 0):
                continue
            Preck_list = []
            for k in range(1, nRelevant + 1):
                test_k = testSet[userID, ranks[userID, -k:]]  # Get labels for the recommended items
                Preck_list.append(np.count_nonzero(test_k) / k / np.count_nonzero(nRelevant))
            avgPrec = np.sum(Preck_list) / nRelevant
            AvgPrec_list.append(avgPrec)
        # Return average Prec@k
        return np.mean(AvgPrec_list)

    def evaluate(self):
        result = self.metric(self.prediction, self.testSet)
        return result


def metrics_evaluate(prediction, matrix_test, get_output=False):
    metrics = ['P@K', 'R@K', 'RPrec', 'AvgP']
    results = []
    for j, metric in enumerate(metrics):
        evaluation = Evaluation(prediction, matrix_test, metric)
        result = evaluation.evaluate()
        print(metric + ': ' + str(round(result, 3)))


class result(object):
    def __init__(self, model_name, dataset, metric, measurement):
        """
        A class for recording each result - help to save into dataframe
        :param model_name: string
        :param dataset: the index of the dataset, int
        :param metric: metric name, string
        :param measurement: metric measurement
        """
        self.model_name = model_name
        self.dataset = dataset
        self.metric = metric
        self.measurement = measurement


def save_results_in_df(results, df_name='', folder_path='results/'):
    """
    Convert the results instance in list to dataframe and save
    :param results: a list of results instance
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if len(df_name) > 0:
        csv_name = folder_path + 'Results_Records_' + df_name +'.csv'
    else:
        csv_name = folder_path + 'Results_Records.csv'
    attributes = results[0].__dict__.keys()
    results_df = pd.DataFrame(columns=attributes, index=range(len(results)))
    for attribute in attributes:
        data = []
        for result in results:
            data.append(result.__getattribute__(attribute))
        results_df[attribute] = data
    results_df.to_csv(csv_name, index=False)


def find_target_measurement(results, target_model_name, target_metric, target_dataset):
    for result in results:
        if ((result.model_name == str(target_model_name)) &
                (result.metric == target_metric) &
                (result.dataset == target_dataset)):
            return result.measurement
    print('the target result cannot be found with ' + str(target_model_name) + ', ' + target_metric +
          ', and dataset no.' + str(target_dataset))


def save_evaluation_in_df(results, models, metrics, n_data, x_label='Model', results_name='', include_time=True, folder_path='results/'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if len(results_name) > 0:
        csv_name = folder_path + 'Results_Evaluation_' + results_name +'.csv'
    else:
        csv_name = folder_path + 'Results_Evaluation.csv'

    metrics_copy = metrics.copy()
    if include_time:
        metrics_copy.append('time')
    evaluation_df = pd.DataFrame(data=models, columns=[x_label], index=range(len(models)))

    for metric in metrics_copy:
        metric_means = []
        metric_lhs = []  # left hand side of the confidence interval
        metric_rhs = []  # right hand side of the confidence interval
        for model in models:
            measurements = np.zeros(n_data)
            for i in range(n_data):
                measurements[i] = find_target_measurement(results, model, metric, i)
            metric_means.append(np.mean(measurements))
            ci = compute_confidence_interval(measurements)
            metric_lhs.append(ci[0])
            metric_rhs.append(ci[1])
        evaluation_df[metric + ' - lower CI'] = metric_lhs
        evaluation_df[metric + ' - mean'] = metric_means
        evaluation_df[metric + ' - upper CI'] = metric_rhs
    evaluation_df.to_csv(csv_name, index=False)


def plot_evaluation(results, models, metrics, n_data, x_label='Models', results_name=''):
    def get_metrics_measurements(results, models, metrics, n_data):
        metrics_measurements = np.zeros((len(metrics), len(models)))
        for i, metric in enumerate(metrics):
            for j, model in enumerate(models):
                measurements = np.zeros(n_data)
                for data in range(n_data):
                    measurements[data] = find_target_measurement(results, model, metric, data)
                metrics_measurements[i, j] = np.mean(measurements)
        return metrics_measurements

    if len(results_name) > 0:
        plot_name = 'Results_Evaluation_' + results_name
    else:
        plot_name = 'Results_Evaluation'

    # Plot of metrics measurements of all models
    img_save_path = plot_name + '_Metrics.png'
    metrics_measurements = get_metrics_measurements(results, models, metrics, n_data)
    plot_multiple_lines(models, metrics_measurements, metrics, x_label, 'Metrics',
                        'Model Performance - Measurements', img_save_path)

    """
    # Plot of computation time of all models
    img_save_path = plot_name + '_Computation_Time.png'
    metrics_measurements = get_metrics_measurements(results, models, ['time'], n_data)
    plot_multiple_lines(models, metrics_measurements, ['time'], 'Models', 'Computation Time',
                        'Model Performance - Computation Time', img_save_path)
    """