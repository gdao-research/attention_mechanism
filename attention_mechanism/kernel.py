import tensorflow as tf


def nonneg_softmax_kernel(data, projection_matrix, is_query, normalize_data=True, eps=1e-4):
    if normalize_data:
        data_normalizer = 1.0/(data.shape[-1]**0.25)
    else:
        data_normalizer = 1.0
    ratio = 1.0/(projection_matrix.shape[0]**0.5)

    data_mod_shape = data.shape[:len(data.shape)-2] + projection_matrix.shape
    data_thick_random_matrix = tf.zeros(data_mod_shape, dtype=projection_matrix.dtype) + projection_matrix

    data_dash = tf.einsum('...id,...jd->...ij', data_normalizer*data, data_thick_random_matrix)

    diag_data = data**2
    diag_data = tf.reduce_sum(diag_data, axis=-1)
    diag_data = (diag_data/2.0)*(data_normalizer**2)
    diag_data = tf.expand_dims(diag_data, axis=-1)

    if is_query:
        data_dash = tf.exp(data_dash - diag_data - tf.reduce_max(data_dash, axis=-1, keepdims=True)) + eps
    else:
        data_dash = tf.exp(data_dash - diag_data - tf.reduce_max(data_dash)) + eps
    return ratio*data_dash

def sincos_softmax_kernel(data, projection_matrix, normalize_data=True):
    if normalize_data:
        data_normalizer = 1.0/(data.shape[-1]**0.25)
    else:
        data_normalizer = 1.0
    ratio = 1.0/(projection_matrix.shape[0]**0.5)

    data_mod_shape = data.shape[:len(data.shape)-2] + projection_matrix.shape
    data_thick_random_matrix = tf.zeros(data_mod_shape, dtype=projection_matrix.dtype) + projection_matrix

    data_dash = tf.einsum('...id,...jd->...ij', data_normalizer*data, data_thick_random_matrix)

    data_dash_sin = ratio*tf.sin(data_dash)
    data_dash_cos = ratio.tf.cos(data_dash)
    data_dash = tf.concat([data_dash_cos, data_dash_sin], axis=-1)

    diag_data = data**2
    diag_data = tf.reduce_sum(diag_data, axis=-1)
    diag_data = (diag_data/2.0)*(data_normalizer**2)
    diag_data = tf.expand_dims(diag_data, axis=-1)

    data_normalizer = tf.reduce_max(diag_data, axis=-2, keepdims=True)
    diag_data -= data_normalizer
    diag_data = tf.exp(diag_data)
    data_prime = data_dash * diag_data
    return data_prime

def generalized_kernel(data, projection_matrix, kernel_fn=tf.nn.relu, eps=0.001, normalize_data=False):
    if normalize_data:
        data_normalizer = 1.0/(data.shape[-1]**0.25)
    else:
        data_normalizer = 1.0

    if projection_matrix is None:
        return kernel_fn(data_normalizer*data) + eps

    data_mod_shape = data.shape[:len(data.shape)-2] + projection_matrix.shape
    data_thick_random_matrix = tf.zeros(data_mod_shape) + projection_matrix

    data_dash = tf.einsum('...id,...jd->...ij', data_normalizer*data, data_thick_random_matrix)
    data_prime = kernel_fn(data_dash) + eps
    return data_prime