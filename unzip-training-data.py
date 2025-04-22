import tarfile
import os
import pandas as pd

PATH_TO_TARFILE = './images_zip'
PATH_TO_IMAGES = './images'

if __name__ == '__main__':
    tar_names = sorted(os.listdir('./images_zip'))
    image_names = set(map(lambda fn: 'images/' + fn, pd.read_csv('base_data.csv')['Image Index']))
    for tar_name in tar_names:
        print(f'Processing {tar_name}')
        tar_fullpath = os.path.join(PATH_TO_TARFILE, tar_name)
        with tarfile.open(tar_fullpath, 'r:gz') as tar:
            available_images = set(tar.getnames())
            available_images &= image_names
            for i, image_name in enumerate(available_images):
                print(f'extracting {image_name} ({i+1}/{len(available_images)})')
                if not os.path.isfile(os.path.join(PATH_TO_IMAGES, image_name)):
                    tar.extract(image_name)