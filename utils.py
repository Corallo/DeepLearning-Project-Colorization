import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from PIL import Image
import glob


ab_bins = np.load('pts_in_hull.npy')
nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', p=2).fit(ab_bins)

def soft_encode_ab(raw_ab):

    raw_ab = raw_ab.numpy()

    # Flatten (C, A, H, W) array into (C*H*W, A) array

    nax = np.setdiff1d(np.arange(0,raw_ab.ndim),np.array((1)))
    axorder = np.concatenate((nax,np.array(1).flatten()),axis=0)

    flat_ab = raw_ab.transpose((axorder)).reshape((-1,2))

    # Calculate encoidings for each element

    distances, indices = nbrs.kneighbors(flat_ab)

    dist_w = np.exp(-distances**2/(2*5**2))
    dist_w = dist_w/np.sum(dist_w,axis=1, keepdims=True)

    encoded_ab_flat = np.zeros((flat_ab.shape[0],ab_bins.shape[0]))
    encoded_ab_flat[np.arange(flat_ab.shape[0])[:,None], indices] = dist_w
    
    # Unflatten (C*H*W, Q) array into (C, Q, H, W)

    reversed_ax = np.argsort(axorder)

    enc_shape = np.array(raw_ab.shape)[nax].tolist()
    enc_shape.append(encoded_ab_flat.shape[1])
    encoded_ab = encoded_ab_flat.reshape(enc_shape).transpose(reversed_ax)
    
    return torch.from_numpy(encoded_ab)


def getYgivenZ(Z, w=64, h=64, Q=313, T=0.38):
    colorsList =ab_bins
    Z=Z.reshape((-1,Q))
    num = np.exp(np.log(Z)/T)
    den = np.sum(np.exp(np.log(Z)/T),axis=1)
    ft= num/den[:,None]
    assert(np.sum(ft,axis=0).all(0)==1) #should sum 1
    Y=np.dot(ft,colorsList).reshape(w,h,2)
    return Y

def generateImg(Z,light):
    Y = getYgivenZ(Z)
    newImg = np.zeros((64,64,3))
    newImg[:,:,0]=light
    newImg[:,:,1:]=Y
    Image = cv2.cvtColor(newImg.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image
    

def testfun():
    img = np.random.randint(0,255,(64,64,3),dtype=np.uint8)

    plt.imshow(img)
    inputImage = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    light = inputImage[:,:,0]
    raw_ab = cv2.resize(inputImage, (64, 64), interpolation = cv2.INTER_AREA)[:,:,1:].astype(float) - 128.0
    raw_ab=raw_ab.reshape(1,2,64,64)
    Z = soft_encode_ab(raw_ab)
    Z = Z[0,:,:,:].transpose(1,2,0)
    Y = getYgivenZ(Z)
    newImg = np.zeros((64,64,3))
    newImg[:,:,0]=light
    newImg[:,:,1:]=Y
    inputImage = cv2.cvtColor(newImg.astype(np.uint8), cv2.COLOR_LAB2RGB)
    plt.imshow(inputImage)
    print(Y)

def load_images(args):
    extensions = ['JPEG', 'jpg', 'png', 'PNG']
    images = []
    for ext in extensions:        
        images += list(glob.glob(os.path.join(args.kmeans_source, "*." + ext)))

    imgs = []
    for path in images:
        img = Image.open(path).convert('RGB').resize((opt.image_size, opt.image_size))
        img = np.array(img).astype(np.float32)
        img = np.expand_dims(img, 0)
        imgs.append(img)
        if len(imgs) > args.batch_size:
            break

    img = np.concatenate(imgs, axis=0)        
    img = img.transpose((0, 3, 1, 2))
    img = img * 2 /255.0 - 1

    return torch.from_numpy(img)