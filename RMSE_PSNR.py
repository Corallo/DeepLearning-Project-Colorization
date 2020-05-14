from math import log10, sqrt 
import cv2 
import numpy as np 
import os
from tqdm import tqdm
from PIL import Image

  
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  
        return (0,100) 
    rmse = sqrt(mse)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / rmse) 
    return rmse, psnr 
  
def main(): 
    n_img = 0
    sumRmse=0
    sumPsnr=0
    file_list = sorted(os.listdir('./img/imagenet-mini/val/'))
    for file in tqdm(file_list[0:]):
        if not (file == '.DS_Store'):
            img_list = sorted(os.listdir('./img/imagenet-mini/val/' +  file))
            for img_path in img_list:
                if not (img_path == '.DS_Store'):
                    path = './img/imagenet-mini/val/' + file + '/' + img_path
                    generatedPath = './img/imagenet-mini/generated/' + file + '/' + img_path
                    original = cv2.imread(path) 
                    compressed = cv2.imread(generatedPath) 
                    rmse, psnr = PSNR(original, compressed) 
                    sumRmse += rmse
                    sumPsnr += psnr
                    n_img+=1
    print(f"avergae RMSE value is {sumRmse/n_img}") 
    print(f"avergae PSNR value is {sumPsnr/n_img} dB") 
       
if __name__ == "__main__": 
    main() 
