# 1. Para poder ignorar capas del tensor de salida se puede hacer np.delete(tensor, [capas ignoradas], axis=2)

## Parece mejor alternativa a 1. usar masked arrays para hacer argmax: https://stackoverflow.com/questions/59745656/find-index-of-max-element-in-numpy-array-excluding-few-indexes

import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor

# Modified from https://github.com/open-mmlab/mmsegmentation/blob/5b33faa1465b823d7560a1e0d69dc4cdaa1d711f/mmseg/models/segmentors/encoder_decoder.py
def simple_test(self, img, img_meta, rescale=True):
    """Simple test with single image."""
    seg_pred = self.inference(img, img_meta, rescale)

    seg_pred = seg_pred.cpu().numpy()
    # unravel batch dim
    seg_pred = list(seg_pred)
    return seg_pred



def tensor_inference(model, img):

    # result is the inference tensor with probabilities per class
    prob_tensor = inference_segmentor(model, img)[0]

    return prob_tensor


def segmentation_inference(model, img):

    prob_tensor = tensor_inference(model, img)

    relevant_classes = np.array([1,2,3,4,5,10,14,18,27,35,47,62,73,77,95,110,141]) - 1
    m = np.ones(prob_tensor.shape, dtype=bool)
    m[relevant_classes] = False

    filtered_result = np.ma.array(prob_tensor, mask=m)

    segmentation_mask = np.argmax(filtered_result, axis=0)

    return segmentation_mask

def model_inference(config_file : str, checkpoint_file : str, imgs_files_list):
    """Builds the model and infers a list of images.

    Args:
        config_file (str): MMSegmentation models config filename.
        checkpoint_file (str): Models .pth checkpoint filename.
        imgs_files_list (list[str]): List of images filenames to infer of.
    Returns:
        list[ndarray]: List of inferred segmentation masks.
    """

    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # Overriding method to instance https://stackoverflow.com/questions/394770/override-a-method-at-instance-level/46757134#46757134
    model.simple_test = simple_test.__get__(model, type(model))

    segmentation_masks = list()
    for img in imgs_files_list:
        print('Inferring:', img)
        segm_mask = segmentation_inference(model, img)
        segmentation_masks.append(segm_mask)

    return segmentation_masks

