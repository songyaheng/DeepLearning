import tensorflow as tf


def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :param x: Input keras network
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and Conv2D added
    :rtype: keras network
    """
    x = tf.keras.layers.BatchNormalization(axis=concat_axis,
                                           gamma_regularizer=tf.keras.regularizers.l2(weight_decay),
                                           beta_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Conv2D(nb_filter, (3, 3),
                               kernel_initializer="he_uniform",
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """
    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = tf.keras.layers.Concatenate(axis=concat_axis)(list_feat)
        nb_filter += growth_rate
    return x, nb_filter


def transition(x, concat_axis, nb_filter,
               dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """

    x = tf.keras.layers.BatchNormalization(axis=concat_axis,
                                           gamma_regularizer=tf.keras.regularizers.l2(weight_decay),
                                           beta_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Conv2D(nb_filter, (1, 1),
                               kernel_initializer="he_uniform",
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

def input_fn(name):
    def featues(name):
        return "{}/images".format(name)
    def labels(name):
        return "{}/labels".format(name)
    train_dataset = tf.data.Dataset.from_tensor_slices((featues(name), labels(name)))
    return train_dataset.batch(32).repeat()

if __name__ == '__main__':
    params={}
    model_input = tf.keras.Input(shape=params["img_dim"])
    assert (params["depth"] - 4) % 3 == 0, "Depth must be 3 N + 4"
    # layers in each dense block
    nb_layers = int((params["depth"] - 4) / 3)
    # Initial convolution
    x = tf.keras.layers.Conv2D(params["nb_filter"], (3, 3),
                               kernel_initializer="he_uniform",
                               padding="same",
                               name="initial_conv2D",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(params["weight_decay"]))(model_input)
    # Add dense blocks
    for block_idx in range(params["nb_dense_block"] - 1):
        x, nb_filter = denseblock(x, params["concat_axis"], nb_layers,
                                  params["nb_filter"], params["growth_rate"],
                                  dropout_rate=params["dropout_rate"],
                                  weight_decay=params["weight_decay"])
        # add transition
        x = transition(x, concat_axis=params["concat_axis"], nb_filter=nb_filter, dropout_rate=params["dropout_rate"],
                       weight_decay=params["weight_decay"])
    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, params["concat_axis"], nb_layers,
                              params["nb_filter"], params["growth_rate"],
                              dropout_rate=params["dropout_rate"],
                              weight_decay=params["weight_decay"])
    x = tf.keras.layers.BatchNormalization(axis=params["concat_axis"],
                                           gamma_regularizer=tf.keras.regularizers.l2(params["weight_decay"]),
                                           beta_regularizer=tf.keras.regularizers.l2(params["weight_decay"]))(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.GlobalAveragePooling2D(data_format=tf.keras.backend.image_data_format())(x)
    x = tf.keras.layers.Dense(params["nb_classes"],
                              activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(params["weight_decay"]),
                              bias_regularizer=tf.keras.regularizers.l2(params["weight_decay"]))(x)
    densenet = tf.keras.models.Model(inputs=[model_input], outputs=[x], name="DenseNet")
    # Model output
    densenet.summary()
    # Build optimizer
    opt = tf.keras.optimizers.Adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    densenet.compile(loss='sparse_categorical_crossentropy',
                     optimizer=opt,
                     metrics=["accuracy"])
    densenet.fit(input_fn("train"), epochs=10, steps_per_epoch=30)
    densenet.evaluate(input_fn("test"))