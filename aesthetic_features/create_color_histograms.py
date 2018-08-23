from src.support_functions import *


def create_color_histograms(images, output_save_path, color_space_name, n_bins=60):

    def preprocess_image(image, color_space_name, n_bins):
        if color_space_name == 'hls':
            img_processed = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            color_range = [[0, 180], [0, 256], [0, 256]]
        elif color_space_name == 'hsv':
            img_processed = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            color_range = [[0, 180], [0, 256], [0, 256]]
        elif color_space_name == 'rgb':
            img_processed = image
            color_range = [[0, 256], [0, 256], [0, 256]]
        else:
            print('Error: the color space is mistaken!')
            return None
        hist_list = []
        for i in range(3):
            hist = cv2.calcHist([img_processed], [i], None, [n_bins], color_range[i]).flatten()
            hist_list.append(hist)
        hist_vector = np.hstack((np.array(hist_list)))
        return hist_vector

    color_histograms = np.zeros((len(images), n_bins * 3))
    for i, image in enumerate(images):
        color_histogram = preprocess_image(image, color_space_name, n_bins)
        color_histograms[i, :] = color_histogram
    save_sparse_csr(output_save_path, csr_matrix(color_histograms))


def main():
    os.chdir('../')
    sample_name_list = ['sample_data_1', 'sample_data_2', 'sample_data_3', 'sample_data_4', 'sample_data_5']
    """
    samples = []
    for sample_name in sample_name_list:
        sample_data_path =  'data/' + sample_name + '/'
        samples.append(SampleData(sample_data_path))
    """
    samples = load_samples(sample_name_list)

    n_bins = 60
    color_space_list = ['rgb', 'hls', 'hsv']
    for color_space_name in color_space_list:
        print()
        print('Color Space: ' + color_space_name)
        for sample_data in samples:
            output_save_path = get_output_path(sample_data.sample_path, color_space_name, n_bins)
            if not os.path.exists(output_save_path):
                images = sample_data.load_photo_contents()
                create_color_histograms(images, output_save_path, color_space_name, n_bins)


if __name__ == "__main__":
    main()