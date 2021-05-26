import os
import json
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

# Imports from https://github.com/intel-isl/DPT
from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from comparison.datasets.eade20k import ExtendedAde20k
from comparison.inference.eval import model_eval, format_models_metrics
from comparison.visualization.visualize import visualize_wrong_pixels, visualize_mask
from comparison.postprocess.pipeline import PostProcessPipeline
from comparison.postprocess.join_classes import JoinClasses

# Some code adapted from https://github.com/intel-isl/DPT/blob/main/run_segmentation.py
def model_inference(model_type, model_path, dataset, keep_classes):

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Device: %s" % device)

    net_w = net_h = 480

    if model_type == "dpt_large":
        model = DPTSegmentationModel(
            150,
            path=model_path,
            backbone="vitl16_384",
        )
    elif model_type == "dpt_hybrid":
        model = DPTSegmentationModel(
            150,
            path=model_path,
            backbone="vitb_rn50_384",
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

    model.eval()

    model.to(device)

    segm_masks = list()
    for img_name, img in zip(dataset.images_filenames, dataset.load_images()):
        print('Inferring:', img_name)
        
        img_input = transform({"image": img / 255.0})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            out = model.forward(sample)

            prediction = torch.nn.functional.interpolate(
                out, size=img.shape[:2], mode="bicubic", align_corners=False
            )

            prediction = prediction.squeeze().cpu().numpy()

        # For masking tensor later
        relevant_classes = np.array(keep_classes) - 1
        m = np.ones(prediction.shape, dtype=bool)
        m[relevant_classes] = False

        filtered_result = np.ma.array(prediction, mask=m)

        segmentation_mask = np.argmax(filtered_result, axis=0)
        segm_masks.append(segmentation_mask)

    return segm_masks


def output_results(dataset, results, outdir, opacity):
    
    pred_dir = os.path.join(outdir, 'predictions')

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for filename, image, result in zip(dataset.ann_filenames, dataset.load_images(), results):
        out_filename = os.path.join(pred_dir, filename)
        print('Saving image results in', out_filename)
        visualize_mask(image, result, dataset.PALETTE, save=out_filename, show=False, opacity=opacity)

def output_metrics(headers, metrics, mean_metrics, dataset, model):
    print('Formatting and saving results metrics')
    out_filename = os.path.join(model['output dir'], model['name'] + '_metrics.csv')
    format_models_metrics(headers, metrics, dataset.images_filenames, mean_metrics, out_filename)

def output_wrong_pixels(dataset, results, outdir):
    pixels_dir = os.path.join(outdir, 'wrong_pixels')

    if not os.path.exists(pixels_dir):
        os.mkdir(pixels_dir)

    for filename, image, result, gt in zip(dataset.ann_filenames, dataset.load_images(), results, dataset.load_annotations()):
        out_filename = os.path.join(pixels_dir, filename)
        print('Saving pixels accuracy image in', out_filename)
        visualize_wrong_pixels(image, gt, result, save=out_filename, show=False)

def apply_postprocess(results, post_process):
    for result in results:
        post_process(result)

"""
    Expected JSON config:
    {
        dataset: {
            images folder: ...,
            ann folder: ...
        },
        models: [
            {
                name: ...,
                config: ...,
                checkpoint: ...,
                output dir: ...
            }, ...
        ]
    }

"""

def main():

    configs_file = os.path.normpath('./configs/config.json')
    opacity = 1

    print('Loading config...')
    with open(configs_file, 'r' ) as conf:
        config = json.load(conf)

    images_dir = os.path.normpath(config['dataset']['images folder'])
    ann_dir = os.path.normpath(config['dataset']['ann folder'])
    
    ## Initialize ann postprocessing pipeline
    vegetation_classes = [5, 10, 18, 73]
    joint_vegetation_class = 151
    
    virtual_classes = ('vegetation',)
    virtual_palette = [[1, 68, 33]]

    ann_post_process = PostProcessPipeline(
        (JoinClasses(classes=vegetation_classes, joint_class=joint_vegetation_class),)
    )

    dataset = ExtendedAde20k(images_dir, ann_dir, ann_post_process, 
                             virtual_classes=virtual_classes, 
                             virtual_palette=virtual_palette)

    ## Initialize pred postprocessing (every index is (ann - 1))
    pred_vegetation_classes = [4, 9, 17, 72]
    pred_joint_vegetation_class = 150

    pred_post_process = PostProcessPipeline(
        (JoinClasses(classes=pred_vegetation_classes, joint_class=pred_joint_vegetation_class),)
    )

    models = config['models']

    keep_classes = [1,2,3,4,5,10,14,18,27,35,47,62,73,77,110,141]

    header_classes = list(keep_classes)
    # removes joint classes
    for removed_class in vegetation_classes:
        header_classes.remove(removed_class)
    # adds the common class
    header_classes.append(joint_vegetation_class)
    keep_classes_index = [class_index - 1 for class_index in header_classes] # Used for indexing

    headers = [dataset.CLASSES[index] for index in keep_classes_index]

    print('Starting models evaluation...')
    for model in models:
        print('#######################################\nEvaluating ', model['name'])
        model['output dir'] = os.path.normpath( model['output dir'] )
        # creates output directory if not exists
        if not (os.path.exists(model['output dir'])):
            os.mkdir(model['output dir'])

        # inference over the dataset images
        results = model_inference(model['config'], model['checkpoint'], dataset, keep_classes)

        apply_postprocess(results, pred_post_process)

        # outputs resulting masks
        output_results(dataset, results, model['output dir'], opacity)

        # computes metrics
        metrics, mean_metrics = model_eval(results, dataset)
        # Filters unused classes metrics (which are nan)
        metrics = [metric[keep_classes_index] for metric in metrics]

        # outputs metrics
        output_metrics(headers, metrics, mean_metrics, dataset, model)

        # output wrong_pixels
        output_wrong_pixels(dataset, results, model['output dir'])
    
    print('Evaluations Done')



if __name__ == '__main__':
    main()