from evaluation.src.parameter_tuning import *
from evaluation.src.support_functions import *

samples = ['sample_data_1', 'sample_data_2', 'sample_data_3', 'sample_data_4']
sample_data_list = load_samples(samples, data_folder='data')

model = 'HSVNN'
metrics = ['P@K', 'R@K', 'RPrec', 'AvgP']
target_param_list = [
    ParameterTuning('sim_method_r', ['cosine'], 'Similarity on Interactions'),
    ParameterTuning('sim_method_p', ['cosine'], 'Similarity on Side Attributes'),
    ParameterTuning('alpha', np.arange(0.01, 0.21, 0.03), 'Weight on the HSV Vectors'),
    ParameterTuning('sim_thresh', np.arange(0.01, 0.42, 0.3), 'Minimum Threshold on Item Similarity measure')
]
parameters_tuning(model, metrics, target_param_list, sample_data_list)