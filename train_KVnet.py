from kvnet_model import *
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



N_BANDS = 8
N_CLASSES = 8  # buildings(grey), roads(black), grass(light green), trees(dark green), bare soil(brown),swimming pool(purple),railway station(yellow) and blue (water)
# CLASS_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.5, 0.7, 0.4, 0.15]     # 0.7 and 0.6
# CLASS_WEIGHTS = [0.1, 0.1, 0.1, 0.4, 0.4, 0.7, 0.55, 0.15]  # 50+30 -- tatti -- 0.0900 training error and validation error = 1.3 to 0.5
CLASS_WEIGHTS = 0.9-np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.7, 0.4, 0.15] )
# CLASS_WEIGHTS = [0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.4, 0.15]  # 30  -- good -- 0.14 -- None

N_EPOCHS = 30
UPCONV = True
PATCH_SZ = 160   # should divide by 16
BATCH_SIZE = 15
TRAIN_SZ = int(2500/10)  # train size
VAL_SZ = int(500/10)    # validation size


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/KV_weightsv4.hdf5'

  # all availiable ids: from "01" to "24"

xtraintot = []
ytraintot = []
xvaltot = []
yvaltot = []

def train_net():
        print("start train net")
        
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(np.array(xtraintot), np.array(ytraintot), batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(np.array(xvaltot),np.array(yvaltot))
                  )
        return model
    

if __name__ == '__main__':
    for imagerange in range(0,3):
        trainIds = [str(i) for i in range(1, 5)]
        print('Reading images')
        for img_id in trainIds:
            X_DICT_TRAIN = dict()
            Y_DICT_TRAIN = dict()
            X_DICT_VALIDATION = dict()
            Y_DICT_VALIDATION = dict()
            img_m = normalize(tiff.imread('./data/kvinputData/res{}.tif'.format(img_id)).transpose([1,2,0]))
            print(img_m.shape)
            mask = tiff.imread('./data/mygt/{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
            print(mask.shape)
            train_xsz = int(2/3 * img_m.shape[0])  # use 75% of image as train and 25% for validation
            X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
            Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
            X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
            Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
            x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
            # print("generating validation patches")
            x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
            xtraintot.extend(x_train)
            ytraintot.extend(y_train)
            xvaltot.extend(x_val)
            yvaltot.extend(y_val)
            
            print(img_id + ' read')
        print('Images were read')
        
        train_net()

    
    
