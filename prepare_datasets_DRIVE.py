import os
import h5py
import numpy as np
from PIL import Image

original_imgs_train = "./DRIVE/training/images/"
groundTruth_imgs_train = "./DRIVE/training/1st_manual/"
borderMasks_imgs_train = "./DRIVE/training/mask/"
original_imgs_test = "./DRIVE/test/images/"
groundTruth_imgs_test = "./DRIVE/test/1st_manual/"
borderMasks_imgs_test = "./DRIVE/test/mask/"

num = 20
channels = 3
height = 584
width = 565
dataset_path = "./DRIVE_datasets/"

if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
    
def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

def get_datasets(image_path,groundTruth_dir,borderMasks_dir,train_test="null"):
    images = np.empty((num,height,width,channels))
    groundTruth = np.empty((num,height,width))
    border_masks = np.empty((num,height,width))
    for path, subdirs, files in os.walk(image_path): 
        for i in range(len(files)):
            img = Image.open(image_path+files[i])
            images[i] = np.asarray(img)
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test=="test":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            else:
                print("ERROR!")
                exit()
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks range correctly.")
    images = np.transpose(images,(0,3,1,2))
    assert(images.shape == (num,channels,height,width))
    groundTruth = np.reshape(groundTruth,(num,1,height,width))
    border_masks = np.reshape(border_masks,(num,1,height,width))
    assert(groundTruth.shape == (num,1,height,width))
    assert(border_masks.shape == (num,1,height,width))
    return images, groundTruth, border_masks

imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
print("saving train images")
write_hdf5(imgs_train, dataset_path + "DRIVE_ori_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DRIVE_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_path + "DRIVE_borderMasks_train.hdf5")

imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
print("saving test images")
write_hdf5(imgs_test,dataset_path + "DRIVE_ori_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DRIVE_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_path + "DRIVE_borderMasks_test.hdf5")

print("process successfully!")