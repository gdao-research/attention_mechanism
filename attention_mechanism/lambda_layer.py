import tensorflow as tf
tfkl = tf.keras.layers


# https://openreview.net/pdf?id=xTJEN-ggl1b
class LambdaLayer(tfkl.Layer):
    def __init__(self, dim_k, n=None, r=None, heads=4, dim_out=None, dim_u=1):
        super(LambdaLayer, self).__init__()
        assert dim_out % heads == 0
        self.dim_out = dim_out
        self.u = dim_u
        self.heads = heads
        self.dim_v = dim_out//heads
        self.dim_k = dim_k

        self.wq = tfkl.Conv2D(dim_k*heads, 1, use_bias=False)
        self.normq = tfkl.BatchNormalization()
        self.wk = tfkl.Conv2D(dim_k*dim_u, 1, use_bias=False)
        self.wv = tfkl.Conv2D(self.dim_v*dim_u, 1, use_bias=False)
        self.normv = tfkl.BatchNormalization()
        self.local_contexts = r is not None
        if self.local_contexts:
            assert (r % 2) == 1, 'Receptive kernel size should be odd number'
            self.pos_conv = tfkl.Conv3D(dim_k, (1, r, r), padding='SAME')
        else:
            assert n is not None
            self.pos_emb = self.add_weight(shape=(n, n, dim_k, dim_u), initializer=tf.keras.initializers.random_normal, trainable=True, name='pos_emb')

    def split(self, x, b, dim1, dim2):
        x = tf.reshape(x, (b, -1, dim1, dim2))
        return tf.transpose(x, perm=[0, 2, 3, 1])

    def call(self, x):
        b, hh, ww, c, u, h = *x.get_shape().as_list(), self.u, self.heads
        q = self.normq(self.wq(x))
        q = self.split(q, b, h, self.dim_k)
        k = self.wk(x)
        k = self.split(k, b, u, self.dim_k)
        v = self.normv(self.wv(x))
        v = self.split(v, b, u, self.dim_v)

        k = tf.nn.softmax(k)
        Lc = tf.einsum('bukm,buvm->bkv', k, v)
        Yc = tf.einsum('bhkn,bkv->bnhv', q, Lc)
        if self.local_contexts:
            v = tf.transpose(tf.reshape(v, (b, u, self.dim_v, hh, ww)), perm=[0, 2, 3, 4, 1])
            Lp = self.pos_conv(v)
            Lp = tf.reshape(tf.transpose(Lp, perm=[0, 1, 4, 2, 3]), (b, self.dim_v, self.dim_k, -1))
            Yp = tf.einsum('bhkn,bvkn->bnhv', q, Lp)
        else:
            Lp = tf.einsum('nmku,buvm->bnkv', self.pos_emb, v)
            Yp = tf.einsum('bhkn,bnkv->bnhv', q, Lp)
        Y = Yc + Yp
        out = tf.reshape(Y, (b, hh, ww, h*self.dim_v))
        return out

    def compute_output_shape(self, input_shape):
        return (*input_shape[:2], self.dim_out)

    def get_config(self):
        config = {'output_dim': (*self.input_shape[:2], self.dim_out)}
        base_config = super(LambdaLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))