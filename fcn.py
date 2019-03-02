from keras import optimizers
from keras.layers import Conv2D,Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Add, Activation,Lambda, ThresholdedReLU,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import random
import cv2
import numpy as np

np.set_printoptions(threshold=np.nan)


train_dir ="\\\\desktop-fjb20b3\\TASF\\ArgosAI-Datasets\\APR\\Training_sets\\APR_FOD_S_V2\\Train\\img"
train_gt ="\\\\desktop-fjb20b3\\TASF\\ArgosAI-Datasets\\APR\\Training_sets\\APR_FOD_S_V2\\Train\\gt"

validation_dir ="\\\\desktop-fjb20b3\\TASF\\ArgosAI-Datasets\\APR\\Training_sets\\APR_FOD_S_V2\\Validation\\img\\0"
validation_gt ="\\\\desktop-fjb20b3\\TASF\\ArgosAI-Datasets\\APR\\Training_sets\\APR_FOD_S_V2\\Validation\\gt\\0"

batch_size = 20


# combine generators into one which yields image and masks
def generator(image_generator, mask_generator):
    while True: yield(next(image_generator), next(mask_generator))

def FCN8( nClasses ,  input_height=192, input_width=192):

    IMAGE_ORDERING =  "channels_last"


    img_input = Input(shape=(input_height, input_width, 1))

    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x=BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x=BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x=BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x=BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x=BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x=BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x=BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x=BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x=BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    x=BatchNormalization()(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)  ## (None, 14, 14, 512)

    pool411 = (Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING))(pool411)

    pool311 = ( Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)

    o = Add(name="add")([pool411_2, pool311])
    o = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Activation('sigmoid'))(o)

    model = Model(img_input, o)

    return model


model = FCN8(nClasses     = 1,
             input_height = 192,
             input_width  = 192)


sgd = optimizers.SGD(lr=0.0001, decay=5**(-4), momentum=0.99, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

datagen_args_train = dict(rescale=1./255,
                    horizontal_flip=True,
                    vertical_flip=True)

train_image_datagen = ImageDataGenerator(**datagen_args_train)
train_mask_datagen = ImageDataGenerator(**datagen_args_train)

#seed = random.randrange(1, 1000)
image_generator = train_image_datagen.flow_from_directory(
    'img',
    target_size=(192, 192),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode=None)

mask_generator = train_mask_datagen.flow_from_directory(
    'gt',
    target_size=(192, 192),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode=None))


model.fit_generator(
    generator(image_generator, mask_generator),
    steps_per_epoch=image_generator.n // batch_size,
    epochs=30)

model.save_weights('my_model_weights.h5')

img = cv2.imread('a.png',0)
img = np.reshape(img,[1,200,200,1])
res_img=model.predict(img)
res_img*=255
res_img=res_img[0]
res_img_rs=np.copy(res_img)
for i1, val in enumerate(res_img):
    for i2,val2 in enumerate(val):
        if(res_img[i1][i2][0]<90):
            res_img_rs[i1][i2][0]=0
        else:
            res_img_rs[i1][i2][0]=255
res_img=res_img_rs
res2_img=np.reshape(res_img,[200,200])
cv2.imwrite("cv.png",res_img.astype('uint8'))
