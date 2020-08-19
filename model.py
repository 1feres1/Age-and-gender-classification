from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense , Flatten ,BatchNormalization, Activation ,Add ,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import Input


def cov_block(X, filters, f, s=2):
    X_shortcut = X
    F1, F2, F3 = filters
    # 1 rst
    X = Conv2D(F1, (1, 1), strides=(s, s) ,)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 2 nd
    X = Conv2D(F2, (f, f), strides=(1, 1), padding="SAME")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 3 rd
    X = Conv2D(F3, (1, 1), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)

    # shortcut
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding="SAME")(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    # adding shortcut
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def identity_block(X, filters, f):
    X_shortcut = X
    F1, F2, F3 = filters

    # 1 rst
    X = Conv2D(F1, (1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 2 nd
    X = Conv2D(F2, (f, f), strides=(1, 1), padding="SAME")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 3 rd
    X = Conv2D(F3, (1, 1), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)

    # shortcut
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    return X


def res_model(input_shape, max_age ,):
    # CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    # -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    X_input = Input(input_shape)

    # zero padding
    X = ZeroPadding2D((3, 3))(X_input)
    # stage1
    X = Conv2D(64, (7, 7), strides=2)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2))(X)

    # stage2
    X = cov_block(X, [64, 64, 256], 3, 1)
    X = identity_block(X, [64, 64, 256], 3)
    X = identity_block(X, [64, 64, 256], 3)
    # stage3
    X = cov_block(X, [128, 128, 512], 3, 2)
    X = identity_block(X, [128, 128, 512], 3)
    X = identity_block(X, [128, 128, 512], 3)
    X = identity_block(X, [128, 128, 512], 3)
    # stage4
    X = cov_block(X, [256, 256, 1024], 3, 2)
    X = identity_block(X, [256, 256, 1024], 3)
    X = identity_block(X, [256, 256, 1024], 3)
    X = identity_block(X, [256, 256, 1024], 3)
    X = identity_block(X, [256, 256, 1024], 3)
    X = identity_block(X, [256, 256, 1024], 3)
    # stage5
    X = cov_block(X, [512, 512, 2048], 3, 2)
    X = identity_block(X, [512, 512, 2048], 3)
    X = identity_block(X, [512, 512, 2048], 3)

    # avragepolling
    # X= AveragePooling2D((2,2))(X)

    X = Flatten()(X)

    X_age = Dense(max_age, activation='sigmoid')(X)
    X_gender = Dense(2, activation='sigmoid')(X)
    #X_expression

    model = Model(inputs=X_input, outputs=[X_gender , X_age])


    return model


model = res_model((412,412,3) , 100)
history = model.summary()









