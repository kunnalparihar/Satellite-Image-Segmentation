# Satellite-Image-Segmentation
Satellite Image Segmentation using Deep Learning.
Satellite images are one of the most powerful and important tools used by the meteorologist. 
They are essentially the eyes in the sky . Today advances in remote sensing technologies have made it possible to capture imagery with resolutions as high as 0.41 metres on earth.
Today’s satellites have the power to ‘see’ and distinguish objects that are as little as 16 inches apart from one another on the ground. 
We have implemented a satellite image classification technique for satellite imagery that classify 8 eight classes namely Water, Grass ,Roads , Building , Trees , Swimming pool , Railway and Bare Soil.
### Input satellite image
![Input](https://raw.githubusercontent.com/kunnalparihar/Satellite-Image-Segmentation/master/satelliteimg.png)
## KV Net
As AutoEncoders are useful for noise filtering.Our UNet-model’s output has noise in it and and also classes that disconnected. Our KV-Net model uses this feature of autoencoders to reconnect the disconnected roads, railways,waters, and etc. which are mostly never disconnected in practice.

### KV Net Input image
![Input](https://raw.githubusercontent.com/kunnalparihar/Satellite-Image-Segmentation/master/KVin.jpg)

### KV Net Output image
![Output](https://raw.githubusercontent.com/kunnalparihar/Satellite-Image-Segmentation/master/KVout.jpg)
      
## Libraries

cv2

tifffile

numpy

keras-gpu

tensorflow-gpu

glob


We have implemented a satellite image classification technique for satellite imagery that classify 8 eight classes namely Water, Grass ,Roads , Building , Trees , Swimming pool , Railway and Bare Soil.

## To run the code

put you sat images in data/sat5band/ folder

run the following lines to train the model for all images(including newly added) again.


python3 edgeGen.py                 # this generates the edge data

python3 water_mask_function.py     # this generates the water data 

python3 Grass_mask_function.py     # this generates the Vegetation data


python3 genpatches.py              # to generate patches for above generated data


python3 train_unet.py              # this will begin the training of unet model


python3 train_kvnet.py             # Training of KV_Net


python3 predict_kvnet.py           # Output will be stored in ./outputs/ of data/test/


Inorder to run it directly using weights saved
download weights from here 
https://drive.google.com/file/d/10xldHiPczByAbWKMJ3Ov0uusbUY5Ve53/view?usp=sharing
https://drive.google.com/file/d/1qXzXAAYm0G6z9SaiLtbc1vEqh3E-qS2N/view?usp=sharing

then run this command

python3 predict.py

python3 predict2.py

python3 predict_kvnet.py


### By - Kunal Parihar , Vivasvan Patel
