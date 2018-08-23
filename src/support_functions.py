import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
from io import BytesIO
from collections import Counter
import scipy.sparse as sparse
from scipy.sparse.linalg import inv
import os, os.path
import time
import random
from scipy.stats import sem, t
from numpy import average, std
from math import sqrt
from scipy.sparse import csr_matrix


class SampleData(object):
    """
    Class to manage all datasets in a given sample folder and create exploratory analysis if needed
    """
    def __init__(self, sample_path, task_name='test'):
        self.sample_path = sample_path
        self.sample_data_path = self._setSampleDataPath(task_name)
        self.matrix_train = self._getSparseMatrix(self.sample_data_path, 'matrix_train')
        self.matrix_test = self._getSparseMatrix(self.sample_data_path, 'matrix_test')
        self.photo_metadata = self._getSparseMatrix(self.sample_path, 'photo_metadata')

    @staticmethod
    def _getSparseMatrix(data_path, matrix_name):
        matrix_path = os.path.join(data_path, matrix_name + '.npz')
        matrix = load_sparse_csr(matrix_path)
        return matrix

    def _setSampleDataPath(self, task_name):
        assert task_name in ['validation', 'test']
        return self.sample_path + task_name + '/'

    def load_photo_hsv(self):
        return load_sparse_csr(self.sample_path + 'hsv_vectors.npz')

    def load_photo_rgb(self):
        return load_sparse_csr(self.sample_path + 'rgb_vectors.npz')

    def load_style_matrix(self):
        return load_sparse_csr(self.sample_path + 'style_vectors.npz')


def load_samples(sample_name_list, data_folder='data', task='test'):
    samples = []
    for sample_name in sample_name_list:
        sample_data_path = data_folder + '/' + sample_name + '/'
        samples.append(SampleData(sample_data_path, task))
    return samples


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def load_image(img_name, size=[300, 300]):
    has_image = True
    try:
        img = Image.open(img_name).convert('RGB')
        img = img.resize(size, Image.ANTIALIAS)
        img = np.array(img)
    except IOError:
        has_image = False
        img = create_random_image(size)
    return img, has_image


def create_random_image(size):
    random_image = np.random.randint(255, size=(size[0], size[1], 3))
    random_img = np.array(random_image, dtype=np.uint8)
    return random_img


def get_output_path(folder_path, color_space_name, n_bins):
    return folder_path + color_space_name + '_' + str(n_bins) + '.npz'


def build_df_photo_attributes(data_photos):
    """
    function to build up the dataframe of the selected attributes in data photos
    selected attributes: keywords, category, editors_chioce
    """
    # keywords to matrix
    occurrence_matrix, topNWords = extract_keywords(data_photos['keywords'], topN=200)
    df_keywords = pd.DataFrame(data=occurrence_matrix.astype(int), columns=topNWords)  # Get keyword matrix

    # categories to matrix
    category_list = data_photos['category'].unique()  # Extract a fixed set of category list
    data_photos['category_controlled'] = data_photos['category'].astype('category', ordered=True,
                                                                        categories=category_list)
    df_categories = pd.get_dummies(data_photos['category_controlled'])

    # editor choice to matrix
    df_editor_choice = data_photos['editors_chioce'].astype(int)

    df_photo_attributes = pd.concat([df_keywords, df_categories, df_editor_choice], axis=1)
    return df_photo_attributes


def extract_keywords(keywords_list, topN = 100):
    """
    Function to extract the top keywords by frequency, and return a matrix of occurrence of top keywords
    input:
    - keywords_list: a list of keywords string from the dataset
    - topN: the number of top keywords to be selected
    output:
    - occurrence_matrix: a matrix of shape (n, topN) that records the occurrence of the top keywords
    - topNWords: the top keywords selected, can be used as a column name of occurrence matrix dataframe
    """
    words_list = []
    for i in range(len(keywords_list)):
        keywords = keywords_list[i]
        if isinstance(keywords, str):
            words = [word.lower() for word in keywords.split('|')]
        else:
            words = ['']
        words_list.append(words)
    labels, freq = compute_keywords_freq(words_list)
    topNWords = labels[:min(topN, len(labels))]
    occurrence_matrix = np.zeros((len(keywords_list),topN))
    for i, topWord in enumerate(topNWords):
        occurrence = ([topWord in words for words in words_list])
        occurrence_matrix[occurrence,i] = 1
    return occurrence_matrix, topNWords


def compute_keywords_freq(words_list):
    counter = Counter()
    for words in words_list:
        counter.update([word for word in words if word != ''])
    labels, values = zip(*counter.items())
    ind_sorted = np.array(values).argsort()[::-1]
    labels = np.array(labels)[ind_sorted]
    values = np.array(values)[ind_sorted]
    return labels, values


def plot_multiple_lines(xs, ys_list, descriptions, xlabel, ylabel, title, img_save_path):
    """
    Function to plot one or more lines in a diagram
    :param xs: a series of x values
    :param ys_list: a list of y values for all the lines
    :param descriptions: corresponding descriptions (label name) for each line in legend
    :param xlabel: name of x-axis
    :param ylabel: name of y-axis
    :param title: title of the diagram
    :return: none; just plot the diagram
    """
    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    markers = ['+', 'o', '*', 'x', 'v', 'd', '^', 's', '>', '<']
    for i, ys in enumerate(ys_list):
        label_stype = colors[i % len(colors)] + markers[i % len(markers)] + '-'
        ax.plot(xs, ys, label_stype, label=descriptions[i])
    # Now add the legend
    ax.legend()
    if len(xs) >= 3:
        if not isinstance(xs[0], str) and xs[0] > 0:
            if ((xs[0] > 0) & (xs[1]/xs[0] == xs[2]/xs[1])):  # The x list is growing exponentially
                plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(img_save_path)


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute the confidence interval of a given data
    :param data: a list of data
    :param confidence:
    :return: confidence interval in list format - (ls, rs)
    """
    return t.interval(confidence, len(data)-1, loc=np.mean(data), scale=sem(data))


def normalize(matrix):
    matrix = matrix.todense()
    output = np.zeros(shape=matrix.shape)
    for i in range(matrix.shape[0]):
        vector = matrix[i,:]
        output[i, :] = (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    return sparse.csr_matrix(output)


