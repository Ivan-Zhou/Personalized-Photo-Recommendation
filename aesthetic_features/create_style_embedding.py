from __future__ import print_function
import argparse
from keras.applications import vgg19
from keras import backend as K
from data.load_data import *


def load_model(input_tensor):
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet', include_top=False)
    return model


# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    output = K.eval(gram)
    return output


def compute_style_of_all_photos(images, output_folder, layer_index):
    for i, image in enumerate(images):
        output_path = output_folder + 'style_' + str(i)
        compute_and_save(images[i], output_path, layer_index)


def compute_and_save(image, photo_output_path, layer_index=0):
    input_tensor = K.variable(image)
    model = load_model(input_tensor)
    style_value_flatten = compute_style(model, layer_index=layer_index).flatten()
    output_csr = csr_matrix(style_value_flatten)
    save_sparse_csr(photo_output_path, output_csr)


def compute_style(model, layer_index=0):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    feature_layers = ['block1_conv1', 'block2_conv1',
                      'block3_conv1', 'block4_conv1',
                      'block5_conv1']

    # for layer_name in feature_layers:
    layer_name = feature_layers[layer_index]
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[0, :, :, :]
    style_value = gram_matrix(style_features)
    return style_value


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_style_of_all_photos(style_folder, style_output_path):
    files = glob(style_folder + '*.npz')
    for i, file in enumerate(files):
        photo_style = load_sparse_csr(file)
        style_matrix = photo_style if i == 0 else np.vstack([style_matrix, photo_style])
    save_sparse_csr(style_output_path, csr_matrix(style_matrix))


def create_style(args):
    os.chdir('../')
    data_folder = args['data_folder']
    layer_index = args['layer_index']
    sample_data_list = get_data(data_folder)
    for sample_data in sample_data_list:
        images = sample_data.load_photo_contents()
        output_vector_path = get_output_path(sample_data.sample_path, 'style_vectors/')
        compute_style_of_all_photos(images, output_vector_path, layer_index)
    output_matrix_path = sample_data.sample_path + 'style_vectors.npz'
    load_style_of_all_photos(output_vector_path, output_matrix_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-folder',
        help='Path to get data folder',
        required=True
    )
    parser.add_argument(
        '--layer_index',
        help='The index of block in VGG19',
        default=3
    )
    args = parser.parse_args()
    create_style(args.__dict__)
