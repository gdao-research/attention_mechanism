import abc
import tensorflow as tf


class RandomMatrix:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError('Abstract method')


class GaussianUnstructuredRandomMatrix(RandomMatrix):
    def __init__(self, nb_rows, nb_cols):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols

    def __call__(self):
        return tf.random.normal((self.nb_rows, self.nb_rows))


class GaussianOrthogonalRandomMatrix(RandomMatrix):
    def __init__(self, nb_rows, nb_cols, scaling=0, qr_uniform_q=False):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.scaling = scaling
        # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
        self.qr_uniform_q = qr_uniform_q

    def _orthogonal_random_matrix(self):
        unstructured_block = tf.random.normal((self.nb_cols, self.nb_cols), dtype=tf.float32)
        q, r = tf.linalg.qr(unstructured_block)
        if self.qr_uniform_q:
            d = tf.linalg.diag_part(r, 0)
            q *= tf.sign(d)
        q = tf.transpose(q)
        return q

    def __call__(self):
        nb_full_blocks = self.nb_rows//self.nb_cols
        block_list = []
        for _ in range(nb_full_blocks):
            q = self._orthogonal_random_matrix()
            block_list.append(q)
        remaining_rows = self.nb_rows - nb_full_blocks*self.nb_cols
        if remaining_rows > 0:
            q = self._orthogonal_random_matrix()
            block_list.append(q[:remaining_rows])
        final_mat = tf.concat(block_list, axis=0)
        if self.scaling == 0:
            multiplier = tf.norm(tf.random.normal((self.nb_rows, self.nb_cols)), axis=1)
        elif self.scaling == 1:
            multiplier = tf.sqrt(tf.cast(self.nb_cols, dtype=tf.float32))*tf.ones((self.nb_rows))
        else:
            raise ValueError(f'Invalid scaling {scaling}. Must be 0 or 1')
        return tf.matmul(tf.linalg.diag(multiplier), final_mat)