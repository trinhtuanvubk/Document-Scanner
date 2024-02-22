import os
import cv2
import shutil
import random
import numpy as np
np.random.seed(42)
random.seed(42)


def generate_doc_set(args):
    DST_IMG_DIR = args.dst_img_dir
    DST_MSK_DIR = args.dst_msk_dir

    os.makedirs(DST_IMG_DIR, exist_ok=True)
    os.makedirs(DST_MSK_DIR, exist_ok=True)

    datasets = {
        args.raw_img_dir: len(os.listdir(raw_img_dir)),  # 2573
    }


    def copy_and_create_mask(img_paths, crop=True):

        for image_path in img_paths:

            img_name = os.path.split(image_path)[-1]
            image = cv2.imread(image_path)

            if crop:
                H, W, _ = image.shape
                image = image[42 : H - 42, 42 : W - 42, :]

            cv2.imwrite(os.path.join(DST_IMG_DIR, img_name), image)

            mask = np.ones_like(image) * 255
            cv2.imwrite(os.path.join(DST_MSK_DIR, img_name), mask)

        return


    for folder_path, total_to_take in datasets.items():
        print(folder_path)

        image_paths = np.asarray([os.path.join(folder_path, i) for i in os.listdir(folder_path)])

        chosen_image_paths = np.random.choice(image_paths, size=total_to_take, replace=False)

        if folder_path == r"DOCUMENTS\\datasets\\docvqa_images":
            crop = True
        else:
            crop = False

        copy_and_create_mask(img_paths=chosen_image_paths, crop=crop)
