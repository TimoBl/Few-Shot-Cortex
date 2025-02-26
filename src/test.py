import numpy as np
import skimage
import scipy
import os
import nibabel as nib

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# measure stats
def get_stats(pred, target, mask=None, path="", label=""):
    
    # we use mask to ignore background
    mask = mask if mask is not None else np.ones_like(pred)
    
    TP = np.nansum(np.logical_and((pred==1) * mask, (target==1) * mask))
    FP = np.nansum(np.logical_and((pred==1) * mask, (target==0) * mask))
    FN = np.nansum(np.logical_and((pred==0) * mask, (target==1) * mask))
    TN = np.nansum(np.logical_and((pred==0) * mask, (target==0) * mask))
    
    stats = {
        "{}/accuracy/{}".format(path, label) : (TP + TN) / (TP + FP + FN + TN),
        "{}/dice/{}".format(path, label) : 2*TP / (2*TP + FP + FN),
        "{}/sensitivity/{}".format(path, label) : TP / (TP + FN),
        "{}/specificity/{}".format(path, label) : TN / (TN + FP)
    }
    return stats

# use to get gaussian filter for essembling
def get_filter(slices, filter_shape):
    w, h = filter_shape // 4
    n = np.zeros(filter_shape)
    n[w:-w, h:-h] = 1
    return np.array([scipy.ndimage.gaussian_filter(n, sigma=10) for s in range(slices)])

# we assume full connectivity of the label, meaning we remove islands
def clean_seg(mask):
    labels = skimage.morphology.label(mask, connectivity=1) # connect
    label, count = np.unique(labels, return_counts=True) # count group
    idx = np.argsort(count)[-2]
    mask = (labels == idx) # choose the second biggest
    return scipy.ndimage.morphology.binary_fill_holes(mask).astype(float)

# saving image
def save_img(img, dst, name, affine):
    mkdir(dst)
    fname = '{}/{}.nii.gz'.format(dst, name)
    niftiImg = nib.Nifti1Image(img.astype(np.float32), affine)
    niftiImg.header['xyzt_units'] = 2  # mm
    nib.save(niftiImg, fname)
    
def mkdir(OUT_DIR):
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

# used to generate hard seg
def post_processing(image, prediction, dst, left_mask=None, right_mask=None, affine=np.eye(4)):
    
    # last label is brian mask 
    brain_prob = prediction[-1]
    brain_mask = (brain_prob >= 0.5)
    
    # soft labels
    gm_prob = prediction[0] * brain_mask
    wm_prob = prediction[1] * brain_mask
    
    # hard segmentation 
    seg_img = (np.argmax([gm_prob, wm_prob], axis=0)+2).astype(np.uint8)
    seg_img[gm_prob+wm_prob < 0.7] = 0

    seg_img[gm_prob > 0.5] = 2
    seg_img[wm_prob > 0.5] = 3
    
    # we clean the segmentation up -> should use victors method as some point
    seg_img = 3 * clean_seg(seg_img == 3) + 2 * clean_seg(seg_img == 2)
    
    # (before thresholding)
    #save_img(brain_prob, dst, 'brainprob', affine)
    save_img(gm_prob, dst, 'gmprob', affine)
    save_img(wm_prob, dst, 'wmprob', affine)

    # thresholding
    # (need to make sure we actually have a gray/white matter interface
    # which is defined as direct neighbouring GMprob > 0.5 with a neighbouring WMprob > 0.5)
    T=0.5

    gm_prob[wm_prob > gm_prob] = 0
    wm_prob[wm_prob > gm_prob] = 1
    wm_prob[seg_img < 1] = 0
    gm_prob[gm_prob > T] = 1
    wm_prob[gm_prob > T] = 0

    save_img(seg_img, dst, 'seg', affine)
    save_img(gm_prob, dst, 'gmprobT', affine)
    save_img(wm_prob, dst,  'wmprobT', affine)
    save_img(image, dst, 'img', affine)
    
    # we add a hemispherical segmentation if the mask is provided (assumes right side)
    if (left_mask is not None) and (right_mask is not None):
        print("Found Hemispherical Mask")
        
        wm = (seg_img == 3)
        gm = (seg_img == 2)
        
        left_hemi = (2 * wm + 3 * gm) * (left_mask) # left wm [2], gm [3]
        right_hemi = (41 * wm + 42 * gm) * (right_mask) # right wm [41], gm [42]
        
        save_img(left_hemi + right_hemi, dst, 'softmax_seg', affine)
        
    return seg_img