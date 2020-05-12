import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from PIL import Image
import glob
import cv2
from torch.nn.functional import interpolate
from torchvision.utils import make_grid
from skimage.color import lab2rgb

ab_bins = np.load('pts_in_hull.npy')
nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', p=2).fit(ab_bins)
ab_bins = torch.from_numpy(ab_bins).cuda()

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

    encoded_ab_flat = np.zeros((flat_ab.shape[0],313))
    encoded_ab_flat[np.arange(flat_ab.shape[0])[:,None], indices] = dist_w
    
    # Unflatten (C*H*W, Q) array into (C, Q, H, W)

    reversed_ax = np.argsort(axorder)

    enc_shape = np.array(raw_ab.shape)[nax].tolist()
    enc_shape.append(encoded_ab_flat.shape[1])
    encoded_ab = encoded_ab_flat.reshape(enc_shape).transpose(reversed_ax)
    
    return torch.from_numpy(encoded_ab)


def decode(Z, T=0.38):

    # New decoder function  
    
    C, Q, H, W = list(Z.size())
    
    flat_Z = softmax(Z/T, dim=1).transpose(1,2).transpose(2,3).reshape((-1, Q))
    flat_Y = torch.mm(flat_Z,ab_bins)
    return flat_Y.reshape((C,H,W,2)).transpose(2,3).transpose(1,2)

def generateImg(Z,light):
    Y = getYgivenZ(Z)
    newImg = np.zeros((64,64,3))
    newImg[:,:,0]=light
    newImg[:,:,1:]=Y
    Image = cv2.cvtColor(newImg.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image

def getImages(L_channel, ab_target, ab_gen, batch_num, decode=True):
    L_channel = interpolate(L_channel[:batch_num,:,:,:] + 50.0, scale_factor=0.25, mode='bilinear', 
        recompute_scale_factor=True, align_corners=True)
    if decode:
        ab_gen = decode(ab_gen[:batch_num,:,:,:], T=0.38)
    else:
        ab_gen = ab_gen[:batch_num,:,:,:]
    
    ab_target = ab_target[:batch_num,:,:,:]
    img_target = torch.cat([L_channel, ab_target], dim=1)
    img_gen = torch.cat([L_channel, ab_gen], dim=1)
    img_all = torch.cat([img_target, img_gen], dim=0).numpy().transpose((0,2,3,1))

    imgs_all_l = []
    for i in range(batch_num):
        imgs_all_l.append(torch.from_numpy(lab2rgb(img_all[i]).transpose((2,0,1))))
        imgs_all_l.append(torch.from_numpy(lab2rgb(img_all[i + batch_num]).transpose((2,0,1))))
    img_rgb = torch.stack(imgs_all_l, dim=0)

    # return all images in a single batch grid
    return make_grid(img_rgb)


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