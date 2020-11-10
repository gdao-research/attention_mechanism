import tensorflow as tf
from attention_mechanism.attention import SelfAttention
from attention_mechanism.sequence import ReversibleSequence, SequentialSequence
from attention_mechanism.utils import ReZero, Normalization, Chunk, FeedForward
tfkl = tf.keras.layers


# https://arxiv.org/pdf/2009.14794v1.pdf
class Performer(tf.keras.Model):
    def __init__(self, dim, depth, heads=8, nb_features=None, feature_type='ortho', redraw_projection=True, ortho_scaling=0.0, qr_uniform_q=False, causal=False, kernel_type='generalized', kernel_fn=tf.nn.relu, ff_rate=0., attn_rate=0., ff_mult=4, reversible=False, ff_chunks=1, use_rezero=False, norm_type='layer', ff_glu=False):
        super(Performer, self).__init__()
        wrapper = ReZero if use_rezero else Normalization
        blocks = []
        for _ in range(depth):
            f = wrapper(SelfAttention(dim, heads=heads, nb_features=nb_features, feature_type=feature_type, redraw_projection=redraw_projection, ortho_scaling=ortho_scaling, qr_uniform_q=qr_uniform_q, causal=causal, kernel_type=kernel_type, kernel_fn=kernel_fn, rate=attn_rate), norm_type)
            g = wrapper(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, rate=ff_rate, glu=ff_glu)))
            blocks.append([f, g])
        self.reversible = reversible
        self.fn = ReversibleSequence(blocks) if reversible else SequentialSequence(blocks)

    def call(self, x, training=True):
        return self.fn(x, training=training)

    def backward_grads_and_vars(self, x, y, dy, training=True):
        # only use when using ReversibleSequence
        dx, grads_all, vars_all = self.fn.backward_grads_and_vars(x, y, dy, training=training)
        return dx, grads_all, vars_all

    @tf.function
    def to_train(self, x, y, loss_fn, training=True, **kwargs):
        if not self.reversible:
            with tf.GradientTape() as tape:
                logit = self(x, training=training)
                loss = loss_fn(y, logit, **kwargs)
            grads_all = tape.gradient(loss, self.trainable_variables)
            vars_all = self.trainable_variables
        else:
            with tf.GradientTape() as tape:
                logit = self(x, training=training)
                loss = loss_fn(y, logit, **kwargs)
            dy = tape.gradient(loss, logit)
            _, grads_all, vars_all = self.backward_grads_and_vars(x, logit, dy, training=training)
        return zip(grads_all, vars_all)


class PerformerLM(tf.keras.Model):
    def __init__(self, n_tokens, max_seq_len, dim, depth, heads=8, nb_features=None, feature_type='ortho', redraw_projection=True, ortho_scaling=0.0, qr_uniform_q=False, causal=False, kernel_type='generalized', kernel_fn=tf.nn.relu, ff_rate=0., attn_rate=0., ff_mult=4, reversible=False, ff_chunks=1, use_rezero=False, norm_type='layer', ff_glu=False, emb_rate=0.):
        super(PerformerLM, self).__init__()
        self.reversible = reversible
        self.max_seq_len = max_seq_len
        self.token_emb = tfkl.Embedding(n_tokens, dim)
        self.pos_emb = tfkl.Embedding(max_seq_len, dim)
        self.dropout = tfkl.Dropout(emb_rate)
        self.performer = Performer(dim, depth, heads, nb_features, feature_type, redraw_projection, ortho_scaling, qr_uniform_q, causal, kernel_type, kernel_fn, ff_rate, attn_rate, ff_mult, reversible, ff_chunks, use_rezero, norm_type, ff_glu)
        self.norm = tfkl.LayerNormalization()
        self.dense = tfkl.Dense(n_tokens)

    def embed(self, x, training=True):
        b, n = x.shape
        x = self.token_emb(x)
        x += self.pos_emb(tf.range(n))
        out = self.dropout(x, training=training)
        return out

    def final_out(self, x):
        norm_out = self.norm(x)
        logit = self.dense(norm_out)
        return logit

    def call(self, x, training=True):
        # x: language 2D -- B, N
        x = self.embed(x, training=training)
        performer_out = self.performer(x, training=training)
        norm_out = self.norm(x)
        logit = self.dense(norm_out)
        return logit

    @tf.function
    def to_train(self, x, y, loss_fn, training=True, **kwargs):
        if not self.reversible:
            with tf.GradientTape() as tape:
                logit = self(x, training=training)
                loss = loss_fn(y, logit, **kwargs)
            grads_all = tape.gradient(loss, self.trainable_variables)
            vars_all = self.trainable_variables
        else:
            grads_all = []
            vars_all = []
            with tf.GradientTape(persistent=True) as tape:
                embed_out = self.embed(x, training=training)
                per_out = self.performer(embed_out, training=training)
                logit = self.final_out(per_out)
                loss = loss_fn(y, logit, **kwargs)
            grads = tape.gradient(loss, [per_out] + self.norm.trainable_variables + self.dense.trainable_variables)
            dper_out = grads[0]
            grads_all += grads[1:]
            vars_all += self.norm.trainable_variables + self.dense.trainable_variables
            dembed_out, grads_per_out, vars_per_out = self.performer.backward_grads_and_vars(embed_out, per_out, dper_out, training=training)
            grads_all += grads_per_out
            vars_all += vars_per_out
            rest_grads = tape.gradient(embed_out, self.token_emb.trainable_variables + self.pos_emb.trainable_variables, output_gradients=dembed_out)
            grads_all += rest_grads
            vars_all += self.token_emb.trainable_variables + self.pos_emb.trainable_variables
        return zip(grads_all, vars_all)