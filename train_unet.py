from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x



N_BANDS = 4 
N_CLASSES = 8  # Brown - Bare Soil, Light green - Grass, Gray - Building, Purple - Swimming Pool, Dark Green - Trees, Black - Roads , Yellow - Railway Station and Blue - Water
CLASS_WEIGHTS = [0.1, 0.1, 0.1, 0.4, 0.4, 0.7, 0.55, 0.15]  # 50+30 -- tatti -- 0.0900 training error and validation error = 1.3 to 0.5
# CLASS_WEIGHTS = [ 6.49350649 , 0.87642419 , 0.275558  , 10 ,  0.70921986  ,0.39651071 , 6.02409639 , 1.05374078]   # not tried but have a positive feeling about these weights
# CLASS_WEIGHTS = [ 0.11619276 , 0.01575265 , 0.0049555 , 0.71596023, 0.01274994, 0.00713148, 0.10830608, 0.01895136]  # calulated v1 good
# CLASS_WEIGHTS = [ 0.11619276 , 0.1575265 , 0.049555 , 0.71596023, 0.1274994, 0.0713148, 0.40830608, 0.1895136]  # calculated_v2  epochsran =40 good but road and railway problem
# CLASS_WEIGHTS = [0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.4, 0.15]  # 30  -- good -- 0.14 -- None

N_EPOCHS = 40 
UPCONV = True
PATCH_SZ = 16*8   # should divide by 16
BATCH_SIZE = 100
TRAIN_SZ = 2500  # train size
VAL_SZ = 500    # validation size

def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weightscalculated_v2.hdf5'


def train_net():
    print("start train net")
    x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
    x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
    model = get_model()
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
    #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
    tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
              verbose=2, shuffle=True,
              callbacks=[model_checkpoint, csv_logger, tensorboard],
              validation_data=(x_val, y_val))
    return model

trainIds = [str(i) for i in range(1, 15)]

if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()
    print("start train net")
    model = get_model()
    print(weights_path)
    # if os.path.isfile(weights_path):
        # model.load_weights(weights_path)
            
    for img_id in trainIds:
        img_m = normalize(tiff.imread('./data/sat/{}.tif'.format(img_id)))
        mask = tiff.imread('./data/mygt/{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
        train_xsz = int(2/3 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
        print('Images were read')
    train_net()

    
    
    
