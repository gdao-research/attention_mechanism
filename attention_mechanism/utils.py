from functools import partial
import math
import tensorflow as tf
from attention_mechanism.random_matrix import GaussianUnstructuredRandomMatrix, GaussianOrthogonalRandomMatrix
from attention_mechanism.kernel import nonneg_softmax_kernel, sincos_softmax_kernel, generalized_kernel
tfkl = tf.keras.layers


def linear_attention(q, k , v):
    d_rep = 1./tf.einsum('...nd,...d->...n', q, tf.reduce_sum(k, axis=-2))
    context = tf.einsum('...nd,...ne->...de', k, v)
    out = tf.einsum('...de,...nd,...n->...ne', context, q, d_rep)
    return out

def causal_linear_attention(q, k, v):
    k_cumsum = tf.cumsum(k, axis=-2)
    context = tf.einsum('...nd,...ne->...nde', k, v)
    context = tf.cumsum(context, axis=-3)
    context /= tf.expand_dims(k_cumsum, axis=-1)
    out = tf.einsum('...nde,...nd->...ne', context, q)
    return out

def scaled_dot_prod_attention(q, k, v, mask):
    q_kt = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_attention_logits = q_kt/tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_w = tf.nn.softmax(scaled_attention_logits, axis=-1)
    out = tf.matmul(attention_w, v)
    return out, attention_w


# https://arxiv.org/pdf/2009.14794v1.pdf
class FastAttentionviaLowRankDecomposition(tf.keras.Model):
    def __init__(self, dim_heads, nb_features=None, feature_type='ortho', redraw_projection=True, ortho_scaling=0.0, qr_uniform_q=False, causal=False, kernel_type='generalized', kernel_fn=tf.nn.relu):
        super(FastAttentionviaLowRankDecomposition, self).__init__()
        self.nb_features = nb_features if nb_features else int(dim_heads*math.log(dim_heads))
        self.dim_heads = dim_heads
        self.ortho_scaling = ortho_scaling
        self.redraw_projection = redraw_projection
        self.attention_fn = causal_linear_attention if causal else linear_attention
        self.kernel_type = kernel_type
        self.kernel_fn = kernel_fn

        if feature_type == 'ortho':
            self.matrix_creator = GaussianOrthogonalRandomMatrix(nb_rows=self.nb_features, nb_cols=dim_heads, scaling=ortho_scaling, qr_uniform_q=qr_uniform_q)
        elif feature_type == 'iid':
            self.matrix_creator = GaussianUnstructuredRandomMatrix(nb_rows=self.nb_features, nb_cols=dim_heads)
        elif feature_type == 'deterministic':
            assert kernel_type == 'generalized'
            self.matrix_creator = None
        else:
            raise ValueError(f'Unknown feature type {feature_type}')
        if not self.redraw_projection:
            self.set_projection_matrix()

    def set_projection_matrix(self):
        self.proj_matrix = self.matrix_creator()

    def call(self, q, k, v):
        if self.redraw_projection and self.matrix_creator:
            proj_matrix = self.matrix_creator()
        elif self.matrix_creator:
            proj_matrix = self.proj_matrix
        else:  # self.matrix_creator is None
            assert self.kernel_type == 'generalized'
            proj_matrix = None  # only apply to generalized kernel

        if self.kernel_type == 'generalized':
            create_kernel = partial(generalized_kernel, projection_matrix=proj_matrix, kernel_fn=self.kernel_fn)
            qp, kp = map(create_kernel, (q, k))
        elif self.kernel_type == 'nonneg':
            create_kernel = partial(nonneg_softmax_kernel, projection_matrix=proj_matrix)
            qp = create_kernel(q, is_query=True)
            kp = create_kernel(k, is_query=False)
        elif self.kernel_type == 'sincos':
            create_kernel = partial(sincos_softmax_kernel, projection_matrix=proj_matrix)
            qp, kp = map(create_kernel, (q, k))
        out = self.attention_fn(qp, kp, v)
        return out


# https://arxiv.org/pdf/2001.04451.pdf
class Chunk(tf.keras.Model):
    def __init__(self, chunks, fn, axis=-1):
        super(Chunk, self).__init__()
        self.chunks = chunks
        self.axis = axis
        self.fn = fn

    def call(self, x, training=True):
        chunks = tf.split(x, self.chunks, axis=self.axis)
        return tf.concat([self.fn(c, training=training) for c in chunks], axis=self.axis)


# https://arxiv.org/pdf/2003.04887.pdf
class ReZero(tf.keras.Model):
    def __init__(self, fn, norm_type='rezero'):  # norm_type is for easier coding
        super(ReZero, self).__init__()
        self.fn = fn
        self.g = tf.Variable(0.0, dtype=tf.float32, training=True)

    def call(self, x, training=True):
        return self.fn(x, training=training)*self.g


class ScaleNorm(tfkl.Layer):
    def __init__(self, eps=1e-5, norm_type='scale'):
        super(ScaleNorm, self).__init__()
        self.g = tf.Variable(1.0, dtype=tf.float32, trainable=True)
        self.eps = eps

    def call(self, x):
        norm = tf.norm(x, axis=-1, keepdims=True)
        n = tf.clip_by_value(norm, clip_value_min=self.eps, clip_value_max=tf.reduce_max(norm))
        return x/n*self.g


class Normalization(tf.keras.Model):
    def __init__(self, fn, norm_type='layer'):
        super(Normalization, self).__init__()
        self.fn = fn
        if norm_type == 'layer':
            self.norm = tfkl.LayerNormalization()
        elif norm_type == 'scale':
            self.norm = ScaleNorm()
        else:
            raise ValueError(f'norm type {norm_type} has not been implemeted. Use layer or scale instead')

    def call(self, x, training=True):
        x = self.norm(x)
        return self.fn(x, training=training)


class GELU(tfkl.Layer):
    def __init__(self):
        super(GELU, self).__init__()
        self.PI = tf.constant(math.pi, dtype=tf.float32, name='pi')

    def call(self, x):
        return 0.5*x*(1. + tf.nn.tanh(tf.sqrt(2.0+self.PI)*(x + 0.044715*tf.pow(x, 3))))


class FeedForward(tf.keras.Model):
    def __init__(self, dim, mult=4, rate=0., activation_fn=GELU(), glu=False):
        super(FeedForward, self).__init__()
        self.glu = glu
        self.dense1 = tfkl.Dense(dim*mult*(2 if glu else 1))
        self.activation_fn = activation_fn
        self.drop1 = tfkl.Dropout(rate)
        self.dense2 = tfkl.Dense(dim)

    def call(self, x, training=True):
        if self.glu:
            x, v = tf.split(self.dense1(x), num_or_size_splits=2, axis=-1)
            x = self.activation_fn(x)*v
        else:
            x = self.dense1(x)
            x = self.activation_fn(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)