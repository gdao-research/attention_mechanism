import tensorflow as tf
tfkl = tf.keras.layers


# https://papers.nips.cc/paper/6816-the-reversible-residual-network-backpropagation-without-storing-activations.pdf
class ReversibleBlock(tf.keras.Model):
    def __init__(self, f_block, g_block, axis=-1):
        super(ReversibleBlock, self).__init__()
        self.f = f_block
        self.g = g_block
        self.axis = axis

    def call(self, x, mask=None, training=True):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)
        f_x2 = self.f(x2, mask, training=training)
        y1 = f_x2 + x1
        g_y1 = self.g(y1, mask, training=training)
        y2 = g_y1 + x2
        return tf.concat([y1, y2], axis=self.axis)

    def backward_grads_and_vars(self, y, dy, mask=None, training=True):
        dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self.axis)

        with tf.GradientTape(persistent=True) as tape:
            y = tf.identity(y)
            tape.watch(y)
            y1, y2 = tf.split(y, num_or_size_splits=2, axis=self.axis)
            z1 = y1
            gz1 = self.g(z1, mask, training=training)
            x2 = y2 - gz1
            fx2 = self.f(x2, mask, training=training)
            x1 = z1 - fx2

            grads_combined = tape.gradient(gz1, [z1] + self.g.trainable_variables, output_gradients=dy2)
            dz1 = dy1 + grads_combined[0]
            dg = grads_combined[1:]
            dx1 = dz1

            grads_combined = tape.gradient(fx2, [x2] + self.f.trainable_variables, output_gradients=dz1)
            dx2 = dy2 + grads_combined[0]
            df = grads_combined[1:]
        del tape
        grads = df + dg
        vars_ = self.f.trainable_variables + self.g.trainable_variables
        x = tf.concat([x1, x2], axis=self.axis)
        dx = tf.concat([dx1, dx2], axis=self.axis)
        return x, dx, grads, vars_


# https://arxiv.org/pdf/2001.04451.pdf
class ReversibleSequence(tf.keras.Model):
    def __init__(self, blocks):
        super(ReversibleSequence, self).__init__()
        self.blocks = []
        for (f, g) in blocks:
            self.blocks.append(ReversibleBlock(f, g))

    def call(self, x, mask=None, training=True):
        for block in self.blocks:
            x = block(x, mask, training=training)
        return x

    def backward_grads_and_vars(self, x, y, dy, mask=None, training=True):
        grads_all = []
        vars_all = []
        for i in reversed(range(len(self.blocks))):
            block = self.blocks[i]
            if i == 0:
                with tf.GradientTape() as tape:
                    x = tf.identity(x)
                    tape.watch(x)
                    y = block(x, mask, training=training)
                grads_combined = tape.gradient(y, [x] + block.trainable_variables, output_gradients=dy)
                dx = grads_combined[0]
                grads_all += grads_combined[1:]
                vars_all += block.trainable_variables
            else:
                y, dy, grads, vars_ = block.backward_grads_and_vars(y, dy, mask, training=training)
                grads_all += grads
                vars_all += vars_
        return dx, grads_all, vars_all


class SequentialSequence(tf.keras.Model):  # standard residual
    def __init__(self, blocks):
        super(SequentialSequence, self).__init__()
        self.blocks = blocks

    def call(self, x, mask=None, training=True):
        for (f, g) in self.blocks:
            x = x + f(x, mask, training=training)
            x = x + g(x, mask, training=training)
        return x