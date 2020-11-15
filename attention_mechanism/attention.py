import numpy as np
import tensorflow as tf
from attention_mechanism.utils import FastAttentionviaLowRankDecomposition, scaled_dot_prod_attention
tfkl = tf.keras.layers


class MultiHeadAttention(tfkl.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.depth = d_model//num_heads
        self.wq = tfkl.Dense(d_model)
        self.wk = tfkl.Dense(d_model)
        self.wv = tfkl.Dense(d_model)
        self.dense = tfkl.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention, attention_w = scaled_dot_prod_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        out = self.dense(concat_attention)
        return out, attention_w


class SelfAttention(tf.keras.Model):
    def __init__(self, dim, heads=8, nb_features=None, feature_type='ortho', redraw_projection=True, ortho_scaling=0.0, qr_uniform_q=False, causal=False, kernel_type='generalized', kernel_fn=tf.nn.relu, rate=0.):
        super(SelfAttention, self).__init__()
        assert dim % heads == 0
        self.heads = heads
        self.wq = tfkl.Dense(dim, use_bias=False)
        self.wk = tfkl.Dense(dim, use_bias=False)
        self.wv = tfkl.Dense(dim, use_bias=False)
        self.fast_attention = FastAttentionviaLowRankDecomposition(dim//heads, nb_features, feature_type, redraw_projection, ortho_scaling, qr_uniform_q, causal, kernel_type, kernel_fn)
        self.dense = tfkl.Dense(dim)
        self.dropout = tfkl.Dropout(rate)

    def split_heads(self, x, batch_size, num_heads, depth):
        # flatten attention dims into a vector in the second dim
        # separate dim into num_heads and depth
        x = tf.reshape(x, (batch_size, -1, num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # TODO: add mask to the computation
    def call(self, x, mask=None, training=True):
        # q, k, v: (n+3)dims -- B, L1, L2, ..., Ln, H*D
        # mask: batch of sequence of 0/1 -- 0 is for over length sequence
        q = self.wq(x)
        k = self.wq(x)
        v = self.wv(x)

        qshape = q.shape
        kshape = k.shape
        vshape = v.shape
        # q, k, v: 4dims -- B, H, L, D
        q = self.split_heads(q, qshape[0], self.heads, qshape[-1]//self.heads)
        k = self.split_heads(k, kshape[0], self.heads, kshape[-1]//self.heads)
        v = self.split_heads(v, vshape[0], self.heads, vshape[-1]//self.heads)

        if mask:
            k = k*mask

        out = self.fast_attention(q, k, v)
        # out: (n+3)dims -- B, L1, L2, ..., Ln, H*D
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, vshape)
        out = self.dense(out)
        return self.dropout(out, training=training)