import numpy as np
import random
from .image_handle import load_hdf5
from .pre_processing import my_PreProc

def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
    x_ = x - int(img_w / 2)
    y_ = y - int(img_h / 2)
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0)
    radius = np.sqrt((x_ * x_) + (y_ * y_))
    return radius < R_inside

def extract_ordered(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape) == 4)
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    N_patches_h = img_h // patch_h
    if (img_h % patch_h != 0):
        print(f"Warning: {N_patches_h} patches in height, with {img_h % patch_h} pixels left over")
    N_patches_w = img_w // patch_w
    if (img_w % patch_w != 0):
        print(f"Warning: {N_patches_w} patches in width, with {img_w % patch_w} pixels left over")
    N_patches_tot = (N_patches_h * N_patches_w) * full_imgs.shape[0]
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))

    iter_tot = 0
    for i in range(full_imgs.shape[0]):
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i, :, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w]
                patches[iter_tot] = patch
                iter_tot += 1
    assert (iter_tot == N_patches_tot)
    return patches

def get_data_training(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV):
    train_imgs_original = load_hdf5(DRIVE_train_imgs_original)
    train_masks = load_hdf5(DRIVE_train_groudTruth)
    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks / 255.
    train_imgs = train_imgs[:, :, 9:574, :]
    train_masks = train_masks[:, :, 9:574, :]
    data_consistency_check(train_imgs, train_masks)
    assert (np.min(train_masks) == 0 and np.max(train_masks) == 1)
    patches_imgs_train, patches_masks_train = extract_random(train_imgs, train_masks, patch_height, patch_width, N_subimgs, inside_FOV)
    data_consistency_check(patches_imgs_train, patches_masks_train)

    return patches_imgs_train, patches_masks_train

def paint_border(data, patch_h, patch_w):
    assert (len(data.shape) == 4)
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    img_h = data.shape[2]
    img_w = data.shape[3]
    new_img_h = ((img_h + patch_h - 1) // patch_h) * patch_h
    new_img_w = ((img_w + patch_w - 1) // patch_w) * patch_w
    new_data = np.zeros((data.shape[0], data.shape[1], new_img_h, new_img_w))
    new_data[:, :, 0:img_h, 0:img_w] = data[:, :, :, :]
    return new_data

def extract_random(full_imgs, full_masks, patch_h, patch_w, N_patches, inside=True):
    if (N_patches % full_imgs.shape[0] != 0):
        print("N_patches must be a multiple of 20")
        exit()
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)
    assert (full_masks.shape[1] == 1)
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((N_patches, full_imgs.shape[1], patch_h, patch_w))
    patches_masks = np.empty((N_patches, full_masks.shape[1], patch_h, patch_w))
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    patch_per_img = N_patches // full_imgs.shape[0]
    iter_tot = 0
    for i in range(full_imgs.shape[0]):
        k = 0
        while k < patch_per_img:
            x_center = random.randint(0 + int(patch_w / 2), img_w - int(patch_w / 2))
            y_center = random.randint(0 + int(patch_h / 2), img_h - int(patch_h / 2))
            if inside and not is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h):
                continue
            patch = full_imgs[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2), x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
            patch_mask = full_masks[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2), x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
            iter_tot += 1
            k += 1
    return patches, patches_masks

def get_data_testing(DRIVE_test_imgs_original, DRIVE_test_groudTruth, Imgs_to_test, patch_height, patch_width):
    test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)
    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks / 255.
    test_imgs = test_imgs[0:Imgs_to_test, :, :, :]
    test_masks = test_masks[0:Imgs_to_test, :, :, :]
    test_imgs = paint_border(test_imgs, patch_height, patch_width)
    test_masks = paint_border(test_masks, patch_height, patch_width)

    data_consistency_check(test_imgs, test_masks)
    assert (np.max(test_masks) == 1 and np.min(test_masks) == 0)
    patches_imgs_test = extract_ordered(test_imgs, patch_height, patch_width)
    patches_masks_test = extract_ordered(test_masks, patch_height, patch_width)
    data_consistency_check(patches_imgs_test, patches_masks_test)
    return patches_imgs_test, patches_masks_test

def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)
    N_patches_img = ((img_h - patch_h) // stride_h + 1) * ((img_w - patch_w) // stride_w + 1)
    N_patches_tot = N_patches_img * full_imgs.shape[0]
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))
    iter_tot = 0
    for i in range(full_imgs.shape[0]):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                patch = full_imgs[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w]
                patches[iter_tot] = patch
                iter_tot += 1
    assert (iter_tot == N_patches_tot)
    return patches

def data_consistency_check(imgs, masks):
    assert (len(imgs.shape) == len(masks.shape))
    assert (imgs.shape[0] == masks.shape[0])
    assert (imgs.shape[2] == masks.shape[2])
    assert (imgs.shape[3] == masks.shape[3])
    assert (masks.shape[1] == 1)
    assert (imgs.shape[1] == 1 or imgs.shape[1] == 3)

def get_data_testing_overlap(DRIVE_test_imgs_original, DRIVE_test_groudTruth, Imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    test_masks = load_hdf5(DRIVE_test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks / 255.
    test_imgs = test_imgs[0:Imgs_to_test, :, :, :]
    test_masks = test_masks[0:Imgs_to_test, :, :, :]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    assert (np.max(test_masks) == 1 and np.min(test_masks) == 0)
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_masks

def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    leftover_h = (img_h - patch_h) % stride_h
    leftover_w = (img_w - patch_w) % stride_w
    if (leftover_h != 0):
        print(f"Height is incompatible with stride {stride_h}: img_h={img_h}, patch_h={patch_h}, leftover={leftover_h}, padding {stride_h - leftover_h} pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_h + (stride_h - leftover_h), img_w))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):
        print(f"Width is incompatible with stride {stride_w}: img_w={img_w}, patch_w={patch_w}, leftover={leftover_w}, padding {stride_w - leftover_w} pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], img_w + (stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:full_imgs.shape[2], 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print(f"New full images shape: {full_imgs.shape}")
    return full_imgs

def recompone(data, N_h, N_w):
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    assert (len(data.shape) == 4)
    N_pacth_per_img = N_w * N_h
    assert (data.shape[0] % N_pacth_per_img == 0)
    N_full_imgs = data.shape[0] // N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    full_recomp = np.empty((N_full_imgs, data.shape[1], N_h * patch_h, N_w * patch_w))
    k = 0
    s = 0
    while (s < data.shape[0]):
        single_recon = np.empty((data.shape[1], N_h * patch_h, N_w * patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w] = data[s]
                s += 1
        full_recomp[k] = single_recon
        k += 1
    assert (k == N_full_imgs)
    return full_recomp

def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape) == 4)
    assert (preds.shape[1] == 1 or preds.shape[1] == 3)
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img
    full_prob = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0
    for i in range(N_full_imgs):
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                full_prob[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += preds[k]
                full_sum[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += 1
                k += 1
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)
    final_avg = full_prob / full_sum
    assert (np.max(final_avg) <= 1.0)
    assert (np.min(final_avg) >= 0.0)
    return final_avg

def pred_only_FOV(data_imgs, data_masks, original_imgs_border_masks):
    assert (len(data_imgs.shape) == 4 and len(data_masks.shape) == 4)
    assert (data_imgs.shape[0] == data_masks.shape[0])
    assert (data_imgs.shape[2] == data_masks.shape[2])
    assert (data_imgs.shape[3] == data_masks.shape[3])
    assert (data_imgs.shape[1] == 1 and data_masks.shape[1] == 1)
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i, x, y, original_imgs_border_masks):
                    new_pred_imgs.append(data_imgs[i, :, y, x])
                    new_pred_masks.append(data_masks[i, :, y, x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

def kill_border(data, original_imgs_border_masks):
    assert (len(data.shape) == 4)
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):
        for x in range(width):
            for y in range(height):
                if not inside_FOV_DRIVE(i, x, y, original_imgs_border_masks):
                    data[i, :, y, x] = 0.0

def inside_FOV_DRIVE(i, x, y, DRIVE_masks):
    assert (len(DRIVE_masks.shape) == 4)
    assert (DRIVE_masks.shape[1] == 1)
    if (x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]):
        return False
    return DRIVE_masks[i, 0, y, x] > 0