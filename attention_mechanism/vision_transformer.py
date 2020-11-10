import tensorflow as tf
from attention_mechanism.attention import MultiHeadAttention
from attention_mechanism.utils import GELU
tfkl = tf.keras.layers
tfki = tf.keras.initializers


class AddPositionalEmb(tfkl.Layer):
    def __init__(self, posemb_init=None):
        super(AddPositionalEmb, self).__init__()
        if posemb_init is None:
            self.posemb_init = tfki.RandomNormal(stddev=0.02)
        else:
            self.posemb_init = posemb_init

    def build(self, input_shape):
        n, p, d = input_shape  # batch, patch, dim
        # trainable positional embedding
        self.pe = self.add_weight(shape=(1, p, d), initializer=self.posemb_init, trainable=True)

    def call(self, x, x_pos=None):
        assert len(x.shape) == 3
        if x_pos is None:
            return x + self.pe
        return x + tf.gather(self.pe[0], x_pos, axis=0)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(AddPositionalEmb, self).get_config()
        config = {'output_dim': (self.input_shape)}
        return dict(list(base_config.intems()) + list(config.items()))


class MlpBlock(tf.keras.Model):
    def __init__(self, mlp_dim, rate=0.1):
        super(MlpBlock, self).__init__()
        self.dense1 = tfkl.Dense(mlp_dim, activation=GELU())
        self.dropout1 = tfkl.Dropout(rate)
        self.dense2 = tfkl.Dense(mlp_dim, activation=GELU())
        self.dropout2 = tfkl.Dropout(rate)

    def call(self, x, training=True):
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        out = self.dropout2(x, training=training)
        return out


class Encoder1DBlock(tf.keras.Model):
    def __init__(self, d_model, num_heads, mlp_dim, rate=0.1):
        super(Encoder1DBlock, self).__init__()
        self.norm1 = tfkl.LayerNormalization()
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads)
        self.drop1 = tfkl.Dropout(rate)
        self.norm2 = tfkl.LayerNormalization()
        self.mlp = MlpBlock(mlp_dim, rate)

    def call(self, inps, mask=None, training=True):
        x = self.norm1(inps)
        x, attn = self.multi_head_attn(x, x, x, mask)
        x = self.drop1(x, training=training)
        x = x + inps
        x = self.norm2(x)
        out = self.mlp(x, training=training)
        return out


class Encoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, mlp_dim, rate=0.1, posemb_init=None):
        super(Encoder, self).__init__()
        self.pos_emb = AddPositionalEmb(posemb_init)
        self.drop1 = tfkl.Dropout(rate)
        self.enc_blocks = []
        for n in range(num_layers):
            self.enc_blocks.append(Encoder1DBlock(d_model, num_heads, mlp_dim))
        self.norm = tfkl.LayerNormalization()

    def call(self, x, x_pos=None, mask=None, training=True):
        x = self.pos_emb(x, x_pos)
        x = self.drop1(x, training=training)
        for lyr in self.enc_blocks:
            x = lyr(x, mask, training)
        return self.norm(x)


class VisionTransformer(tf.keras.Model):
    def __init__(self, n_class, patch_size, representation_size, num_layers, num_heads, mlp_dim, d_model=None, hidden_size=None, classifier='gap', batch_size=32, n_channels=3, rate=0.1, posemb_init=None, use_encoder=True):
        super(VisionTransformer, self).__init__()
        self.n_class = n_class
        self.patch_size = patch_size  # patch_size: 2 elements tuple/list
        if d_model is None:
            d_model = patch_size[0]*patch_size[1]*n_channels
        self.representation_size = representation_size
        self.classifier = classifier
        if hidden_size:
            self.conv = tfkl.Conv2D(hidden_size, (patch_size[0], patch_size[1]), strides=(patch_size[0], patch_size[1]), valid='VALID')
        else:
            self.conv = None
        self._zeros = tf.zeros((1, 1, n_channels))
        self._zeros = tf.tile(self._zeros, (batch_size, 1, 1))
        self.encoder = Encoder(num_layers, d_model, num_heads, mlp_dim, rate, posemb_init) if use_encoder else None
        if representation_size is not None:
            self.dense = tfkl.Dense(representation_size, activation='tanh')
        else:
            self.dense = None
        self.final = tfkl.Dense(n_class)

    def call(self, x, x_pos=None, mask=None, training=True):
        n, h, w, c = x.shape
        fh, fw = self.patch_size
        gh, gw = h//fh, w//fw
        if self.conv:
            x = self.conv(x)
        else:  # linear projection of flatten patches
            x = tf.reshape(x, (n, gh, fh, gw, fw, c))
            x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
            x = tf.reshape(x, (n, gh, gw, -1))

        n, h, w, c = x.shape
        x = tf.reshape(x, (n, h*w, c))
        if self.classifier == 'token':
            x = tf.concat([self._zeros, x], axis=1)
        if self.encoder:
            x = self.encoder(x, x_pos, mask, training=training)
        if self.classifier == 'token':
            x = x[:, 0]
        elif self.classifier == 'gap':
            x = tf.reduce_mean(x, axis=list(range(1, tf.rank(x)-1)))
        if self.dense:
            x = self.dense(x)
        return self.final(x)