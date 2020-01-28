import tensorflow as tf


def gated_linear_layer(inputs, gates, name=None):
    activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)

    return activation


def instance_norm_layer(
        inputs,
        epsilon=1e-06,
        activation_fn=None,
        name=None):
    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs=inputs,
        epsilon=epsilon,
        activation_fn=activation_fn)

    return instance_norm_layer


def conv1d_layer(
        inputs,
        filters,
        kernel_size,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=None,
        name=None):
    conv_layer = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)

    return conv_layer


def conv2d_layer(
        inputs,
        filters,
        kernel_size,
        strides,
        padding='same',
        activation=None,
        kernel_initializer=None,
        name=None):
    conv_layer = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)

    return conv_layer


def residual1d_block(
        inputs,
        filters=1024,
        kernel_size=3,
        strides=1,
        name_prefix='residule_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs=h1_glu, filters=filters // 2, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs=h2, activation_fn=None, name=name_prefix + 'h2_norm')

    h3 = inputs + h2_norm

    return h3


def downsample1d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        name_prefix='downsample1d_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def downsample2d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        name_prefix='downsample2d_block_'):
    h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def upsample1d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        shuffle_size=2,
        name_prefix='upsample1d_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs=h1, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs=h1_shuffle, activation_fn=None, name=name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs=h1_gates, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def upsample2d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        shuffle_size=2,
        name_prefix='upsample2d_block_'):
    h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_shuffle = tf.depth_to_space(input=h1, block_size=2, name='h1_shuffle')
    h1_norm = instance_norm_layer(inputs=h1_shuffle, activation_fn=None, name=name_prefix + 'h1_norm')

    h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_shuffle_gates = tf.depth_to_space(input=h1_gates, block_size=2, name='h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def pixel_shuffler(inputs, shuffle_size=2, name=None):
    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor=inputs, shape=[n, ow, oc], name=name)

    return outputs


def atr_concat(inputs, vec):
    num_domains =  vec.get_shape().as_list()[1]

    l = tf.reshape(vec,[-1,1,1,num_domains])
    b = tf.shape(inputs)[0]
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    k = tf.ones([b, h, w, num_domains])
    k = k * l
    x = tf.concat([inputs, k],axis=3)
    return x

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])
        s = tf.shape(x)
        y = tf.reshape(x, [group_size, -1, num_new_features, s[1] // num_new_features, s[2], s[3]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)
        y = tf.reduce_mean(y, axis=2)
        y = tf.cast(y, x.dtype)

        y = tf.tile(y, [group_size, s[1], s[2], 1])
        return tf.concat([x, y], axis=-1)

def Generator212(inputs, vec, num_domains, dim=24, batch_size=1, reuse=False, scope_name='generator_212'):
    # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    l = tf.reshape(vec,[-1,1,1,num_domains])
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    k = tf.ones([batch_size,h,w,num_domains])
    k = k * l
    inputs = tf.concat([inputs, k], axis=3)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs=inputs, filters=128, kernel_size=[5, 15], strides=[1, 1], activation=None,
                          name='h1_conv')
        h1_gates = conv2d_layer(inputs=inputs, filters=128, kernel_size=[5, 15], strides=[1, 1], activation=None,
                                name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name='h1_glu')

        d1 = downsample2d_block(inputs=h1_glu, filters=256, kernel_size=[5, 5], strides=[2, 2],
                                name_prefix='downsample2d_block1_')
        d2 = downsample2d_block(inputs=d1, filters=256, kernel_size=[5, 5], strides=[2, 2],
                                name_prefix='downsample2d_block2_')

        d3 = tf.squeeze(tf.reshape(d2, shape=(batch_size, 1, -1, 2304)), axis=[1], name='d2_reshape')
        resh1 = conv1d_layer(inputs=d3, filters=256, kernel_size=1, strides=1, activation=None, name='resh1_conv')
        resh1_norm = instance_norm_layer(inputs=resh1, activation_fn=None, name='resh1_norm')

        r1 = residual1d_block(inputs=resh1_norm, filters=512, kernel_size=3, strides=1, name_prefix='res1d_block1_')
        r2 = residual1d_block(inputs=r1, filters=512, kernel_size=3, strides=1, name_prefix='res1d_block2_')
        r3 = residual1d_block(inputs=r2, filters=512, kernel_size=3, strides=1, name_prefix='res1d_block3_')
        r4 = residual1d_block(inputs=r3, filters=512, kernel_size=3, strides=1, name_prefix='res1d_block4_')
        r5 = residual1d_block(inputs=r4, filters=512, kernel_size=3, strides=1, name_prefix='res1d_block5_')
        r6 = residual1d_block(inputs=r5, filters=512, kernel_size=3, strides=1, name_prefix='res1d_block6_')

        resh2 = conv1d_layer(inputs=r6, filters=2304, kernel_size=1, strides=1, activation=None, name='resh2_conv')
        resh2_norm = instance_norm_layer(inputs=resh2, activation_fn=None, name='resh2_norm')
        resh3 = tf.reshape(tf.expand_dims(resh2_norm, axis=1), shape=(batch_size, 9, -1, 256), name='resh2_reshape')

        # Upsample
        #resh3 = atr_concat(resh3, vec)
        u1 = upsample2d_block(inputs=resh3, filters=1024, kernel_size=5, strides=1, shuffle_size=2,
                              name_prefix='upsample2d_block1_')
        #u1 = atr_concat(u1, vec)
        u2 = upsample2d_block(inputs=u1, filters=512, kernel_size=5, strides=1, shuffle_size=2,
                              name_prefix='upsample2d_block2_')

        #u2 = atr_concat(u2, vec)
        conv_out = conv2d_layer(inputs=u2, filters=1, kernel_size=[5, 15], strides=[1, 1], activation=None,
                                name='conv_out')
        out = tf.squeeze(conv_out, axis=[-1], name='out_squeeze')

        #out = tf.tanh(out)
        #out = tf.clip_by_value(out, -1+1e-12, 1-1e-12)


    return out


def PatchGanDiscriminator(inputs_A, inputs_B, vec, num_domains, reuse=[False, False], scope_name='discriminator', method='adversarial'):
    # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]

    def feature_layer(inputs):
        with tf.variable_scope(scope_name + '_feature_layer') as scope:
            # Discriminator would be reused in CycleGAN
            if reuse[0]:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

            inputs = tf.expand_dims(inputs, -1)

            h1 = conv2d_layer(inputs=inputs, filters=128, kernel_size=[3, 3], strides=[1, 1], activation=None,
                              name='h1_conv')
            h1_gates = conv2d_layer(inputs=inputs, filters=128, kernel_size=[3, 3], strides=[1, 1], activation=None,
                                    name='h1_conv_gates')
            h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name='h1_glu')

            d1 = downsample2d_block(inputs=h1_glu, filters=256, kernel_size=[3, 3], strides=[2, 2],
                                    name_prefix='downsample2d_block1_')
            d2 = downsample2d_block(inputs=d1, filters=512, kernel_size=[3, 3], strides=[2, 2],
                                    name_prefix='downsample2d_block2_')
            d3 = downsample2d_block(inputs=d2, filters=1024, kernel_size=[3, 3], strides=[2, 2],
                                    name_prefix='downsample2d_block3_')
            d4 = downsample2d_block(inputs=d3, filters=1024, kernel_size=[1, 5], strides=[1, 1],
                                    name_prefix='downsample2d_block4_')

            #out = conv2d_layer(inputs=d4, filters=1, kernel_size=[1, 3], strides=[1, 1], activation=None,
            #                   name='out_conv')

            return d4

    if method == 'adversarial':
        f = feature_layer(inputs_A)
        with tf.variable_scope(scope_name + '_adversarial') as scope:
            if reuse[1]:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

            #f = minibatch_stddev_layer(f)
            adv = conv2d_layer(inputs=f, filters=1, kernel_size=[1, 3], strides=[1, 1], activation=None,
                               name='out_conv_adv')
            return adv

    if method == 'interpolation':
        f = feature_layer(inputs_A)
        with tf.variable_scope(scope_name + '_interpolation') as scope:
            if reuse[1]:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

            i0 = conv2d_layer(inputs=f, filters=128, kernel_size=[3, 3], strides=[1, 1], activation=None,
                               name='int1_conv')
            #i0_gates = conv2d_layer(inputs=f, filters=128, kernel_size=[3, 3], strides=[1, 1], activation=None,
            #                   name='int1_conv_gates')
            #i0_glu = gated_linear_layer(inputs=i0, gates=i0_gates, name='int1_glu')
            interp = tf.reduce_mean(i0, axis=3, keepdims=True)
            #interp = conv2d_layer(inputs=i0_glu, filters=1, kernel_size=[1, 1], strides=[1, 1], activation=None,
            #                   name='out_conv_int')
            return interp

    if method == 'matching':
        assert inputs_B != None
        f1 = feature_layer(inputs_A)
        f2 = feature_layer(inputs_B)
        with tf.variable_scope(scope_name + '_matching') as scope:
            if reuse[1]:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

            l = tf.reshape(vec,[-1,1,1,num_domains])
            b = tf.shape(f1)[0]
            h = tf.shape(f1)[1]
            w = tf.shape(f1)[2]
            k = tf.ones([b,h,w,num_domains])
            k = k * l
            m0 = tf.concat([f1, f2, k], axis=3)
            m1 = conv2d_layer(inputs=m0, filters=1024, kernel_size=[3, 3], strides=[1, 1], activation=None,
                              name='mat1_conv')
            m1_gates = conv2d_layer(inputs=m0, filters=1024, kernel_size=[3, 3], strides=[1, 1], activation=None,
                                    name='mat1_conv_gates')
            m1_glu = gated_linear_layer(inputs=m1, gates=m1_gates, name='mat1_glu')

            mat = conv2d_layer(inputs=m1_glu, filters=1, kernel_size=[1, 3], strides=[1, 1], activation=None,
                               name='out_conv_mat')
            return mat
