from sklearn.metrics.pairwise import pairwise_distances
from src.support_functions import *
from scipy import sparse
from scipy.sparse import vstack
from scipy.stats import pearsonr


class RecSys(object):
    def __init__(self, model, data, latent_k=500):
        self.model_name = model
        self.model = self._getModel(self.model_name)
        self.data = data
        self.latent_k = latent_k  # Number of dimensions in latent factors

    def _getSimMethod(self, sim_method_name):
        switcher = {
            'cosine': self.cosine,
            'euclidean': self.euclidean,
            'jaccard': self.jaccard,
            'pearson': self.pearson,
            '': '',
        }
        return switcher[sim_method_name]

    def _getModel(self, model_name):
        switcher = {
            'ItemAverage': self.itemAverage,
            'Random': self.randomrec,
            'ItemSim': self.itemcf,
            'ItemSimWithInfo': self.itembasedrec,
            'HSVNN': self.hsvnn,
            'RGBNN': self.rgbnn,
            'StyleNN': self.stylenn,
        }
        return switcher[model_name]

    def _setDataItems(self, data_items):
        self.data_items = data_items

    @staticmethod
    def cosine(matrix):
        """
        cosine similarity - cosine pairwise_distance supports sparse matrix as an input
        """
        similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        # this is exactly the same as cosine_similarity from sklearn
        return similarity_matrix

    @staticmethod
    def euclidean(matrix):
        """
        euclidean similarity - euclidean pairwise_distance supports sparse matrix as an input
        """
        distance = pairwise_distances(matrix, metric='euclidean')  # sqrt distance
        similarity_matrix = np.divide(1, 1 + distance)
        return similarity_matrix

    @staticmethod
    def jaccard(matrix):
        matrix = matrix.toarray()
        matrix = matrix.astype(bool)  # Convert to boolean for jaccard distance calculation
        similarity_matrix = np.nan_to_num(1 - pairwise_distances(matrix, metric='jaccard'))
        return similarity_matrix

    @staticmethod
    def pearson(matrix):
        """
        Pearson Correlation: p_corr = E[(X-mu_x)*(Y-mu_y)]/(sigma_x*sigma_y)
        :param matrix: sparse matrix
        :return: correlation matrix
        """
        matrix = matrix.toarray()
        # Rowwise mean of input arrays & subtract from input arrays themselves
        m_mean = matrix - matrix.mean(1)[:, None]
        # Sum of squares across rows
        ssM = (m_mean ** 2).sum(1);
        # Finally get corr coeff
        return np.dot(m_mean, m_mean.T) / np.sqrt(np.dot(ssM[:, None], ssM[None]))

    @staticmethod
    def pearsonr(matrix):
        """
        Implementation based on pearsonr in scipy and use for loop
        :param matrix:
        :return:
        """
        matrix = matrix.toarray()
        n = matrix.shape[0]
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                corr = pearsonr(matrix[i, :], matrix[j, :])
        return corr

    @staticmethod
    def cf_with_item_attributes(interaction, item_attributes, sim_method_r, sim_method_p, alpha, sim_thresh):
        sim_r = sim_method_r(interaction.T)
        sim_p = sim_method_p(item_attributes)
        similarity = alpha * sim_p + (1-alpha) * sim_r
        similarity[similarity < sim_thresh] = 0
        similarity = sparse.csr_matrix(similarity)
        prediction = similarity.dot(interaction.T).T
        prediction[interaction.nonzero()] = 0
        return prediction

    def itemcf(self, sim_method, sim_thresh=0.1):
        """
        Predict with the Item-based Collaborative Filtering
        """
        self.sim_method = self._getSimMethod(sim_method)
        matrix_train = self.data.matrix_train
        ii_similarity = self.sim_method(matrix_train.T)
        ii_similarity[ii_similarity < sim_thresh] = 0  # Remove the impact of dissimilar entities
        ii_similarity = sparse.csr_matrix(ii_similarity)
        prediction = ii_similarity.dot(matrix_train.T).T
        prediction[matrix_train.nonzero()] = 0  # Remove the prediction on already rated items
        return prediction

    def itembasedrec(self, sim_method_r, sim_method_p, alpha=0.1, sim_thresh=0.01):
        """
        Item-based Nearest Neighbour Method with Side Information of Items
        """
        self.sim_method_r = self._getSimMethod(sim_method_r)
        self.sim_method_p = self._getSimMethod(sim_method_p)
        matrix_train = self.data.matrix_train
        item_attributes = self.data.photo_metadata
        prediction = self.cf_with_item_attributes(matrix_train, item_attributes, self.sim_method_r, self.sim_method_p,
                                             alpha, sim_thresh)
        return prediction

    def hsvnn(self, sim_method_r, sim_method_p, alpha=0.01, sim_thresh=0.31):
        self.sim_method_r = self._getSimMethod(sim_method_r)
        self.sim_method_p = self._getSimMethod(sim_method_p)
        matrix_train = self.data.matrix_train
        hsv_vectors = self.data.load_photo_hsv()
        prediction = self.cf_with_item_attributes(matrix_train, hsv_vectors, self.sim_method_r, self.sim_method_p,
                                             alpha, sim_thresh)
        return prediction

    def rgbnn(self, sim_method_r, sim_method_p, alpha=0.01, sim_thresh=0.31):
        self.sim_method_r = self._getSimMethod(sim_method_r)
        self.sim_method_p = self._getSimMethod(sim_method_p)
        matrix_train = self.data.matrix_train
        hsv_vectors = self.data.load_photo_rgb()
        prediction = self.cf_with_item_attributes(matrix_train, hsv_vectors, self.sim_method_r, self.sim_method_p,
                                                  alpha, sim_thresh)
        return prediction

    def stylenn(self, sim_method_r, sim_method_p, alpha=0.01, sim_thresh=0.31):
        self.sim_method_r = self._getSimMethod(sim_method_r)
        self.sim_method_p = self._getSimMethod(sim_method_p)
        matrix_train = self.data.matrix_train
        hsv_vectors = self.data.load_style_matrix()
        prediction = self.cf_with_item_attributes(matrix_train, hsv_vectors, self.sim_method_r, self.sim_method_p,
                                                  alpha, sim_thresh)
        return prediction

    def itemAverage(self):
        matrix_train = self.data.matrix_train
        item_mean = np.mean(matrix_train, axis=0)[0]
        prediction = np.tile(item_mean, (matrix_train.shape[0], 1))
        prediction[matrix_train.nonzero()] = 0  # Remove the prediction on already rated items
        prediction = sparse.csr_matrix(prediction)
        return prediction

    def randomrec(self):
        matrix_train = self.data.matrix_train
        prediction = np.random.rand(matrix_train.shape[0], matrix_train.shape[1])
        prediction[matrix_train.nonzero()] = 0  # Remove the prediction on already rated items
        prediction = sparse.csr_matrix(prediction)
        return prediction

    def predict_all(self, **kwargs):
        self.__pred = self.model(**kwargs)

    def getPrediction(self):
        return self.__pred

    def getPredColName(self):
        return self.pred_column_name
