import albumentations as A
import matplotlib.pyplot as plt
import cv2

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])


import matplotlib.pyplot as plt
def cv2_imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

image = cv2.imread("lung_mri.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2_imshow(image)

#transform will return a dictionary with a single key image. 
#Value at that key will contain an augmented image.
transformed = transform(image=image)
transformed_image = transformed["image"]
cv2_imshow(transformed_image)

import os
import random
import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import label2rgb


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

#function to visualize bounding box
#The visualization function is based on 
#https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2, **kwargs):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img
#function to display title text
def visualize_titles(img, bbox, title, font_thickness = 2, font_scale=0.35, **kwargs):
    x_min, y_min = bbox[:2]
    x_min = int(x_min)
    y_min = int(y_min)
    ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 
                                                     font_scale, font_thickness)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), 
                  (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR,
                font_thickness, lineType=cv2.LINE_AA)
    return img

#function to apply transforms and display transformed images
def augment_and_show(aug, image, mask=None, bboxes=[], categories=[], 
                     category_id_to_name=[], filename=None,
                     font_scale_orig=0.35, font_scale_aug=0.35, show_title=True, **kwargs):

    if mask is None:
        augmented = aug(image=image, bboxes=bboxes, category_ids=categories)
    else:
        augmented = aug(image=image, mask=mask, bboxes=bboxes, category_ids=categories)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB)

    for bbox in bboxes:
        visualize_bbox(image, bbox, **kwargs)

    for bbox in augmented['bboxes']:
        visualize_bbox(image_aug, bbox, **kwargs)

    if show_title:
        for bbox,cat_id in zip(bboxes, categories):
            visualize_titles(image, bbox, category_id_to_name[cat_id],
                             font_scale=font_scale_orig, **kwargs)
        for bbox,cat_id in zip(augmented['bboxes'], augmented['category_ids']):
            visualize_titles(image_aug, bbox, 
                             category_id_to_name[cat_id], font_scale=font_scale_aug, **kwargs)


    if mask is None:
        f, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].imshow(image)
        ax[0].set_title('Original image')

        ax[1].imshow(image_aug)
        ax[1].set_title('Augmented image')
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 16))

        if len(mask.shape) != 3:
            mask = label2rgb(mask, bg_label=0)
            mask_aug = label2rgb(augmented['mask'], bg_label=0)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask_aug = cv2.cvtColor(augmented['mask'], cv2.COLOR_BGR2RGB)

        ax[0, 0].imshow(image)
        ax[0, 0].set_title('Original image')

        ax[0, 1].imshow(image_aug)
        ax[0, 1].set_title('Augmented image')

        ax[1, 0].imshow(mask, interpolation='nearest')
        ax[1, 0].set_title('Original mask')

        ax[1, 1].imshow(mask_aug, interpolation='nearest')
        ax[1, 1].set_title('Augmented mask')

    f.tight_layout()

    if filename is not None:
        f.savefig(filename)

    if mask is None:
        return augmented['image'], None, augmented['bboxes']

    return augmented['image'], augmented['mask'], augmented['bboxes']

#helper function to find a filename in a directory
def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname))]

image = cv2.imread('lung_mri.jpg')


random.seed(42)

bbox_params = A.BboxParams(format='coco', label_fields=['category_ids'])

light = A.Compose([
    A.RandomBrightnessContrast(p=1),
    A.RandomGamma(p=1),
    A.CLAHE(p=1),
], p=1, bbox_params=bbox_params)

medium = A.Compose([
    A.CLAHE(p=1),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
], p=1, bbox_params=bbox_params)


strong = A.Compose([
    A.RGBShift(p=1),
     A.Blur(p=1),
     A.GaussNoise(p=1),
     A.ElasticTransform(p=1),
], p=1, bbox_params=bbox_params)



r = augment_and_show(light, image)
plt.show()

r = augment_and_show(medium, image)
plt.show()

r = augment_and_show(strong, image)
plt.show()