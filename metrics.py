import tensorflow as tf


class ConfusionMatrix(tf.metrics.Metric):

    def __init__(self, num_class):
        super().__init__()
        self.num = num_class
        self.weight = tf.Variable(initial_value=tf.zeros(shape=(num_class, num_class)), trainable=False)

    def update_state(self, y_true, y_pred):
        update = tf.math.confusion_matrix(y_true, y_pred, num_classes=2, dtype=tf.float32)
        self.weight.assign(self.weight + update)

    def result(self):
        return self.weight

    def reset_states(self):
        self.weight.assign(tf.zeros_like(self.weight))


def accuracy():
    return tf.keras.metrics.Accuracy()




if __name__ == '__main__':
    a = ConfusionMatrix(num_class=2)
    acc = Accuracy()
    b = tf.random.uniform(shape=(5, 2), maxval=1., dtype=tf.float32)
    c = tf.random.uniform(shape=(5,), maxval=2, dtype=tf.int32)
    b = tf.argmax(b, axis=1)
    b = tf.cast(b, dtype=tf.int32)
    print(b)
    print(c)
    a.update_state(b,c)
    acc.update_state(b, c)
    print(a.result())
    print(acc.result())
    d = tf.random.uniform(shape=(5, 2), maxval=1., dtype=tf.float32)
    e = tf.random.uniform(shape=(5,), maxval=2, dtype=tf.int32)
    d = tf.argmax(d, axis=1)
    d = tf.cast(d, dtype=tf.int32)
    print(d)
    print(e)
    a.update_state(d, e)
    print(a.result())




