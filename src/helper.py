import os
import random
import numpy as np
import time
import scipy
import skimage
import nibabel as nib
import matplotlib.pyplot as plt
import nibabel.processing as nib_processing
import nibabel.orientations as nib_orientations
import skimage.segmentation
from skimage.segmentation import flood, flood_fill

np.set_printoptions(precision=3)


def reorient(src_nib, orientation="PIL"):
    start_ornt = nib_orientations.io_orientation(src_nib.affine)
    end_ornt = nib_orientations.axcodes2ornt(orientation)
    transform = nib_orientations.ornt_transform(start_ornt, end_ornt)
    return src_nib.as_reoriented(transform)

def save_img(img, dst, name, affine):
    mkdir(dst)
    fname = '{}/{}.nii.gz'.format(dst, name)
    niftiImg = nib.Nifti1Image(img.astype(np.float32), affine)
    niftiImg.header['xyzt_units'] = 2  # mm
    nib.save(niftiImg, fname)

def save_metrics(tracker, metrics, epoch):
    out = {}
    for key in metrics[0].keys():
        l = [item[key] for item in metrics]
        tracker.add_scalar(key, np.mean(l, axis=0), epoch) # we mean the metrics
    return out


def standardize_image(image, mask=None, lower_threshold=None, upper_threshold=None):
    # if mask is provided we use to mask outside, else larger than 0 
    mask = image > 0 if mask is None else mask > 0
    
    # mri image might have sometimes exploiding values, so we limit this
    lower_threshold = 0 if lower_threshold is None else lower_threshold 
    upper_threshold = np.percentile(image[mask], 99.9) if upper_threshold is None else lower_threshold
    
    image[image < lower_threshold] = lower_threshold
    image[image > upper_threshold] = upper_threshold
    
    # and we standardize the image (ignoring background)
    mean = np.mean(image[mask])
    sd = np.std(image[mask])
    
    return (image - mean) / sd
    

def mkdir(OUT_DIR):
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)


# cropping volume
def get_crop(volume):
    nonempty = np.argwhere(volume)
    top_left = nonempty.min(axis=0)
    bottom_right = nonempty.max(axis=0)
    
    return (top_left, bottom_right)

def apply_crop(volume, crop):
    top_left, bottom_right = crop
    cropped = volume[top_left[0]:bottom_right[0]+1,
                   top_left[1]:bottom_right[1]+1,
                   top_left[2]:bottom_right[2]+1]
    return cropped

def apply_uncrop(template, volume, crop):
    uncropped = np.zeros_like(template).astype(volume.dtype)
    top_left, bottom_right = crop
    
    uncropped[top_left[0]:bottom_right[0]+1,
                   top_left[1]:bottom_right[1]+1,
                   top_left[2]:bottom_right[2]+1] = volume
    return uncropped


# so we crop volume but keep it the same scanner space
def crop_nifti(src_nib, crop=None, threshold=0):
    img, affine = src_nib.get_fdata(), src_nib.affine
    
    if crop is None: # we default to thresholding
        crop = get_crop((img > threshold))
    
    # crop image
    img = apply_crop(img, crop)
    
    # translate
    affine[:3, 3] += np.dot(affine[:3, :3], crop[0])
    return nib.Nifti1Image(img, affine)


# pads outside by half of the window size
def conv_pad(array, shape):
    sx, sy, sz = shape // 2
    return np.pad(array, ((sx, sx), (sy, sy), (sz, sz)))

# depads outside by half of the window size
def conv_depad(array, shape):
    sx, sy, sz = shape // 2
    return array[sx:-sx, sy:-sy, sz:-sz]

# grid indices
def grid_indices(mask, stride=(1, 46, 46), filters=(5, 96, 96)):
    shp = np.array(filters)//2

    # we get the bounds (and make sure we are in limits)
    points = np.array(np.where(mask)).T
    minb = np.maximum(points.min(axis=0), shp)
    maxb = np.minimum(points.max(axis=0), mask.shape - shp)

    # create regular grid
    x, y, z = np.meshgrid(*zip([np.arange(minb[i], maxb[i], stride[i]) for i in range(3)]))
    new_points = np.array([x.flatten(), y.flatten(), z.flatten()]).T

    return new_points


# uses to weight the example by the number of example in filter
def conv_weight(counts, slices, shape):
    weights = np.zeros_like(counts)
    filt = np.ones(shape[1:]) # we filter in 2d
    
    for idx, slic in enumerate(slices):
        weights[slic] = scipy.signal.fftconvolve(counts[slic], filt, mode="same")
    
    return np.clip(weights, 0, weights.max())