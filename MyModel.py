
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K



##### Model #####
def UNet(input_shape,num_classes, LR, nblayers):
    model = tf.keras.Sequential()
    input = tf.keras.Input(input_shape)
    x = input
    output = 64
    skipList = []
    # Encoder
    for i in range(nblayers):
        x = blockConv(x, output*(2**i))
        skipList.append(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
    
    # Bottom of the u-net
    x = tf.keras.layers.Conv2D(output*(2**nblayers), kernel_size=3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(output*(2**nblayers), kernel_size=3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)

    # Decoder
    for i in range(nblayers):
        x = blockUpConv(x, skipList[nblayers-(i+1)], output*(2**(nblayers-i)))


    x = tf.keras.layers.Conv2D(num_classes, (1,1), padding="same")(x)

    l = 'binary_crossentropy'
    outputs = tf.keras.activations.sigmoid(x)
      
    model = tf.keras.Model(input, outputs, name="U-Net")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss = l,
        metrics = ['accuracy',tf.keras.metrics.BinaryIoU(threshold=0.5),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    
    model.summary()
    
    return model



def blockConv(input, output):
    x = tf.keras.layers.Conv2D(output, kernel_size=3, padding="same")(input)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(output, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def blockUpConv(input, skip, output):
    x = tf.keras.layers.Conv2DTranspose(output, 3, 2, padding="same")(input)
    x = tf.keras.layers.concatenate([x, skip])
    x = blockConv(x, output)
    return x
