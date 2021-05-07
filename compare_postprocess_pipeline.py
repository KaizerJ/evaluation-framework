import os
import json

from comparison.datasets.eade20k import ExtendedAde20k
from comparison.inference.inference import model_inference
from comparison.inference.eval import model_eval, format_models_metrics
from comparison.visualization.visualize import visualize_wrong_pixels, visualize_mask
from comparison.postprocess.pipeline import PostProcessPipeline
from comparison.postprocess.join_classes import JoinClasses


def output_results(dataset, results, outdir):
    
    pred_dir = os.path.join(outdir, 'predictions')

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for filename, image, result in zip(dataset.ann_filenames, dataset.load_images(), results):
        out_filename = os.path.join(pred_dir, filename)
        print('Saving image results in', out_filename)
        visualize_mask(image, result, dataset.PALETTE, save=out_filename, show=False)

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

    keep_classes = [1,2,3,4,5,10,14,18,27,35,47,62,73,77,95,110,141]
    keep_classes_index = [class_index - 1 for class_index in keep_classes] # Used for indexing

    headers = [dataset.CLASSES[index] for index in keep_classes_index]

    print('Starting models evaluation...')
    for model in models:
        print('#######################################\nEvaluating ', model['name'])
        model['output dir'] = os.path.normpath( model['output dir'] )
        # creates output directory if not exists
        if not (os.path.exists(model['output dir'])):
            os.mkdir(model['output dir'])

        # inference over the dataset images
        results = model_inference(model['config'], model['checkpoint'], dataset.images_full_filenames, keep_classes)

        apply_postprocess(results, pred_post_process)

        # outputs resulting masks
        output_results(dataset, results, model['output dir'])

        # computes metrics
        metrics, mean_metrics = model_eval(results, dataset, '')
        # Filters unused classes metrics (which are nan)
        metrics = [metric[keep_classes_index] for metric in metrics]

        # outputs metrics
        output_metrics(headers, metrics, mean_metrics, dataset, model)

        # output wrong_pixels
        output_wrong_pixels(dataset, results, model['output dir'])
    
    print('Evaluations Done')


        


if __name__ == '__main__':
    main()