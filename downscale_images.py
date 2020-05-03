import os
from tqdm import tqdm
from PIL import Image

file_list = sorted(os.listdir('ILSVRC2012_img_train'))

for file in tqdm(file_list[264:]):
    if not (file == '.DS_Store'):
        img_list = sorted(os.listdir('ILSVRC2012_img_train/' +  file))
        for img_path in img_list:
            if not (img_path == '.DS_Store'):
                path = 'ILSVRC2012_img_train/' + file + '/' + img_path
                img = Image.open(path).convert('RGB')
                img = img.resize((256,256))
                img.save(path)