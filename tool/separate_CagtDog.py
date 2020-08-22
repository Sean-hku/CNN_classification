import os
import shutil

src_folder = r"C:\Users\hkuit164\Downloads\train\train"
cat_folder = '../data/CatDog/train/cat'
dog_folder = '../data/CatDog/train/dog'

os.makedirs(cat_folder,exist_ok=True)
os.makedirs(dog_folder,exist_ok=True)


for img in os.listdir(src_folder):
    # img = img[:3] + img[4:]
    if "cat" in img:
        # print(os.path.join(cat_folder, img))
        shutil.move(os.path.join(src_folder, img), os.path.join(cat_folder, img))
    elif "dog" in img:
        # print(os.path.join(dog_folder, img))
        shutil.move(os.path.join(src_folder, img), os.path.join(dog_folder, img))

