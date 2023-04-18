import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, input_D=128, out_kernels_1=36, out_kernels_2=48, L=512, K=1):
        super(Attention, self).__init__()
        self.L = L  # 512 node fully connected layer
        self.output_D = ((((input_D - 3) // 2) - 4) // 2) + 1
        self.kernel_1 = out_kernels_1
        self.kernel_2 = out_kernels_2
        self.D = input_D  # 128 node attention layer
        self.K = K

        self.feature_extractor_part1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.kernel_1, kernel_size=4),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2, stride=2),
            tf.keras.layers.Conv2D(self.kernel_2, kernel_size=3),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2, stride=2)
        ])

        self.feature_extractor_part2 = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.L),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.L),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5)
        ])

        self.attention = tf.keras.Sequential([
            tf.keras.layers.Dense(self.D),
            tf.keras.layers.Tanh(),
            tf.keras.layers.Dense(self.K)
        ])

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.L * self.K),
            tf.keras.layers.Sigmoid()
        ])

    def call(self, x):
        x = tf.squeeze(x, 0)

        H = self.feature_extractor_part1(x)
        H = tf.reshape(H, (-1, self.kernel_2 * self.output_D * self.output_D))
        H = self.feature_extractor_part2(H)

        A = self.attention(H)  # NxK
        A = tf.transpose(A, (1, 0))  # KxN
        A = tf.nn.softmax(A, axis=1)  # softmax over N

        M = tf.linalg.matmul(A, H)

        Y_prob = self.classifier(M)
        Y_hat = tf.cast(tf.greater_equal(Y_prob, 0.5), dtype=tf.float32)

        return Y_prob, Y_hat, tf.cast(A, dtype=tf.uint8)

    def calculate_all(self, X, Y):
        Y = tf.cast(Y, dtype=tf.float32)
        Y_prob, Y_hat, A = self.call(X)
        error = 1. - tf.reduce_mean(tf.cast(tf.equal(Y_hat, Y), dtype=tf.float32))
        neg_log_likelihood = -1. * (Y * tf.math.log(Y_prob) + (1. - Y) * tf.math.log(1. - Y_prob))

        return neg_log_likelihood, error, Y_hat, A
