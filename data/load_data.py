import numpy as np
from scipy.sparse import csr_matrix
from PIL import Image
import requests
from io import BytesIO
from glob import glob
from scipy.sparse.linalg import inv
import os, os.path
import time
import random
from numpy import average, std
from math import sqrt


class SampleData(object):
    """
    Class to manage all datasets in a given sample folder and create exploratory analysis if needed
    """
    def __init__(self, sample_path, task_name='test'):
        self.sample_path = sample_path
        self.sample_image_path = os.path.join(sample_path, 'photos/')
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

    def load_photo_contents(self):
        image_path  = glob(self.sample_image_path)
        images = []
        for path in image_path:
            images.append(load_image(path))
        return images

    def load_photo_hsv(self):
        return load_sparse_csr(self.sample_path + 'hsv_vectors.npz')

    def load_photo_rgb(self):
        return load_sparse_csr(self.sample_path + 'rgb_vectors.npz')

    def load_style_matrix(self):
        return load_sparse_csr(self.sample_path + 'style_vectors.npz')


def get_data(data_folder_path, task='test'):
    sample_folder_lists = glob(data_folder_path + '*/')
    samples = []
    for sample_data_path in sample_folder_lists:
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
