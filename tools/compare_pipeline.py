import os
import json

from comparison.datasets.ade20k import ade20k
from comparison.inference.inference import model_inference
from comparison.inference.eval import model_eval, format_models_metrics
from comparison.visualization.visualize import visualize_wrong_pixels, visualize_mask


def output_results(dataset, results, outdir):
    
    pred_dir = os.path.join(outdir, 'predictions')

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for filename, image, result in zip(dataset.images_filenames, dataset.load_images(), results):
        out_filename = os.path.join(pred_dir, filename)
        visualize_mask(image, result, dataset.PALETTE, save=out_filename, show=False)

def output_metrics(headers, metrics, mean_metrics, dataset, model):
    out_filename = os.path.join(model['output dir'], model['name'] + '_metrics.csv')
    format_models_metrics(headers, metrics, dataset.filenames, mean_metrics, out_filename)

def output_wrong_pixels(dataset, results, outdir):
    pixels_dir = os.path.join(outdir, 'wrong_pixels')

    if not os.path.exists(pixels_dir):
        os.mkdir(pixels_dir)

    for filename, image, result, gt in zip(dataset.images_filenames, dataset.load_images(), results, dataset.load_annotations()):
        out_filename = os.path.join(pixels_dir, filename)
        visualize_wrong_pixels(image, gt, result, save=out_filename, show=False)

"""
    Expected JSON config:
    {
        dataset: {
            images folder: ...,
            ann folder: ...
        },
        models: [
            {
                model name: ...,
                config file: ...,
                checkpoint file: ...,
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
    dataset = ade20k(images_dir, ann_dir)

    models = config['models']
    headers = dataset.CLASSES

    print('Starting evaluating models...')
    for model in models:
        print('Evaluating ', model['model name'])
        model['output dir'] = os.path.normpath( model['output dir'] )
        # creates output directory if not exists
        if not (os.path.exists(model['output dir'])):
            os.mkdir(model['output dir'])

        # inference over the dataset images
        results = model_inference(model['config'], model['checkpoint'], dataset.images_filenames)

        # outputs resulting masks
        output_results(dataset, results, model['output dir'])

        # computes metrics
        metrics, mean_metrics = model_eval(results, dataset, '')

        # outputs metrics
        output_metrics(headers, metrics, mean_metrics, dataset, model)

        # output wrong_pixels
        output_wrong_pixels(dataset, results, model['output dir'])
    
    print('Evaluations Done')


        


if __name__ == '__main__':
    main()