import numpy as np
import cv2

# create confution matrix from 2 images.
def confutionmatrix(img,gt):
    
    namespace = {(0,80,150):0,(0,255,0):1,(100,100,100):2,(255,150,150):3,(0,125,0):4,(0,0,0):5,(0,255,255):6,(150,0,0):7,(255,255,255):8}
    cm = np.zeros((9,9),dtype=np.int32)
    #print(cm)

    assert img.shape == gt.shape
    
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            cm[namespace[tuple(gt[i][j])]][namespace[tuple(img[i][j])]] +=1
    
    return(cm)

def kappa(cm):
    #cm = confutionmatrix(img,gt)
    N = np.sum(cm)
    summer = 0
    for i in range(cm.shape[0]):
        summer += (np.sum(cm[:][i]) * np.sum(cm[i][:]) )
    po = cm.trace()/N
    pe = summer/(N*N)
    return((po-pe)/(1-pe))

def f1Score(cm):
    
    #cm = confutionmatrix(img,gt)
    f1 = 0
    for i in range(cm.shape[0]):
        if (np.sum(cm[i][:]) !=0 and np.sum(cm[:][i]) !=0 and cm[i][i] !=0):
            recall = (cm[i][i]/np.sum(cm[i][:]))
            precision = cm[i][i]/np.sum(cm[:][i])
            f1 += 2*recall*precision/(recall+precision)
    return f1/cm.shape[0]

for i in range(1,15):
    img = cv2.imread('/Users/kunal/Desktop/accuracy/train_output/maptestres'+str(i)+'.tif',1)
    print("image no ", i)
    print(img1.shape)
    gt = cv2.imread('/Users/kunal/Desktop/inter_iit_Tech/gt/'+str(i)+'.tif',1)
    print(img2.shape)
    cm1 = confutionmatrix(img,gt)
    print(cm1,"with whi te")
    newcm1 = cm1[0:8,0:8]
    print(newcm1.shape,"without white")
    print(newcm1)
    print(kappa(cm1),"kappa score without white")
    print(kappa(newcm1),"kappa with white")
    print(f1Score(cm1),"f1Score without white")
    print(f1Score(newcm1),"f1score with white")    