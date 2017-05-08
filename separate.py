import cv2,glob,os,sys
import numpy as np
import matplotlib.pyplot as plt

def getCluster(arr,cthres=254.,width=10):
    clus = []
    lastWhite  = 0
    first = None
    for i in range(len(arr)):
        if ( lastWhite < i ) and ( arr[i] >= cthres ):
            clus.append([lastWhite,i])
        if arr[i]>=cthres:
            lastWhite = i+1

    nclus = []
    for c in clus:
        if (c[1]-c[0])>width:
            nclus.append(c)

    if len(nclus)==0:
        nclus = [[0,len(arr)-1]]

    return nclus


def show(path,outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    if type(img)==type(None): return
    if img.ndim==2:
        pass
    elif img.ndim==3:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    elif img.ndim==4:
        mask = img[:,:,3]<255
        img  = img[:,:,0:2]
        img[mask,0:2] = 255

    colorThres = 250.
    widthThres = 10
    aspectThres = 1.5
    minSize    = 64

    count = 0

    y = np.mean(img,axis=1)
    cy = getCluster(y,cthres=colorThres,width=widthThres)
    for c in cy:
        yimg = img[c[0]:c[1],:]
        y = np.mean(yimg,axis=0)
        cx = getCluster(y,cthres=colorThres,width=widthThres)
        for cc in cx:
            xyimg = yimg[:,cc[0]:cc[1]]
            if xyimg.shape[0]>aspectThres*xyimg.shape[1]:continue
            if xyimg.shape[1]>aspectThres*xyimg.shape[0]:continue
            if np.min(xyimg.shape)<minSize: continue
            if xyimg.max()==0.: continue
            
            simg = np.zeros((np.max(xyimg.shape),np.max(xyimg.shape)),dtype=np.float32)
            simg[:,:] = 255.
            y1 = int( 0.5 * ( simg.shape[0] - xyimg.shape[0] ) )
            y2 = y1 + xyimg.shape[0]
            x1 = int( 0.5 * ( simg.shape[1] - xyimg.shape[1] ) )
            x2 = x1 + xyimg.shape[1]
            simg[y1:y2,x1:x2] = xyimg
            newfname = os.path.splitext(os.path.basename(path))[0]
            newfname = newfname +"_%d.png"%count
            cv2.imwrite(os.path.join(outpath,newfname),simg)
            count += 1
            print newfname

baseDir = "img"
sepaDir = "sep"
for dir1 in os.listdir(baseDir):
    print dir1
    for fpath in os.listdir(os.path.join(baseDir,dir1)):
        show(os.path.join(baseDir,dir1,fpath),os.path.join(sepaDir,dir1))
