import os
import h5py
import numpy as np
from PIL import Image
import cv2

def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

original_train = "./CHASEDB1/training/images/"
groundTruth_train = "./CHASEDB1/training/labels1/"
original_test = "./CHASEDB1/test/images/"
groundTruth_test = "./CHASEDB1/test/labels1/"

N_train = 20
N_test = 8
channels = 3
height = 960
width = 999
dataset_path = "./CHASEDB1_datasets/"

if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)


def generate_mask(img_array):
    gray = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def get_datasets(imgs_dir, groundTruth_dir, n_imgs):
    imgs = np.empty((n_imgs, height, width, channels))
    groundTruth = np.empty((n_imgs, height, width))
    border_masks = np.empty((n_imgs, height, width))

    for path, subdirs, files in os.walk(imgs_dir):
        files.sort()
        for i in range(len(files)):
            if i >= n_imgs:
                break
            img = Image.open(imgs_dir + files[i])
            img_array = np.asarray(img)
            imgs[i] = img_array

            groundTruth_name = files[i].replace('.jpg', '_1stHO.png')
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)

            mask = generate_mask(img_array)
            border_masks[i] = mask
    assert (np.min(groundTruth) == 0 and np.min(border_masks) == 0)
    print("ground truth and border masks range correctly.")

    imgs = np.transpose(imgs, (0, 3, 1, 2))
    assert (imgs.shape == (n_imgs, channels, height, width))
    groundTruth = np.reshape(groundTruth, (n_imgs, 1, height, width))
    border_masks = np.reshape(border_masks, (n_imgs, 1, height, width))
    assert (groundTruth.shape == (n_imgs, 1, height, width))
    assert (border_masks.shape == (n_imgs, 1, height, width))
    return imgs, groundTruth, border_masks

imgs_train, groundTruth_train, border_masks_train = get_datasets(original_train,groundTruth_train,N_train)

print("saving train images")
write_hdf5(imgs_train, dataset_path + "CHASEDB1_ori_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "CHASEDB1_groundTruth_train.hdf5")
write_hdf5(border_masks_train, dataset_path + "CHASEDB1_borderMasks_train.hdf5")

imgs_test, groundTruth_test, border_masks_test = get_datasets(original_test,groundTruth_test,N_test)
print("saving test images")
write_hdf5(imgs_test, dataset_path + "CHASEDB1_ori_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "CHASEDB1_groundTruth_test.hdf5")
write_hdf5(border_masks_test, dataset_path + "CHASEDB1_borderMasks_test.hdf5")

print("process successfully!")