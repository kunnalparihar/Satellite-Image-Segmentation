import numpy as np
import argparse
import cv2
import matplotlib
import math
from matplotlib import pyplot as plt
import gdal

ds = gdal.Open("/Users/kunal/Desktop/sat_test/"+str(p)+".tif")
band1 = ds.GetRasterBand(1).ReadAsArray()    #Red
band2 = ds.GetRasterBand(2).ReadAsArray()    # Green 
band3 = ds.GetRasterBand(3).ReadAsArray()    #blue
band4 = ds.GetRasterBand(4).ReadAsArray()
band5 = np.zeros((band1.shape[0],band1.shape[1],3))
band5[:,:,0] = band4/np.max(band4)
band5[:,:,1] = band2/np.max(band2) - band5[:,:,0] 
band5[:,:,2] = band3/np.max(band3) - band5[:,:,0] 

band6 = np.zeros((band1.shape[0],band1.shape[1]))
for i in range(band1.shape[0]):
    for j in range(band1.shape[1]):
        if ((band5[i][j][1] <= 0 and band5[i][j][1] >= -0.26) and (band5[i][j][2] <= 0 and band5[i][j][2] >= -0.27)):
            band6[i][j] = 1
        else:
            band6[i][j] = 0

            
print(band6.shape, " shape of grass mask band 6")
plt.imsave("/Users/kunal/Desktop/Test_Grass_Mask/"+str(p)+".png",band6,cmap='gray')
print(" Grass mask Saved")