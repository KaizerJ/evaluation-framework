import os
from collections import OrderedDict
import csv

from comparison.metrics.metrics import eval_metrics
from comparison.datasets.base import BaseDataset

def get_full_filenames(input_dir):
    filenames = os.listdir(input_dir)
    full_filenames = list()
    for filename in filenames:
        full_filenames.append(os.path.join(input_dir, filename))
    
    return full_filenames

def model_eval(results: list, dataset : BaseDataset, output_dir):
    """Evaluates a models results.
    
    Args:
        config_file (str): MMSegmentation models config filename.
        checkpoint_file (str): Models .pth checkpoint filename.
        input_dir (str): Input images directory.
        output_dir (str): Results outputs directory.
    Returns:
        list[ndarray]: List of inferred segmentation masks.
    """

    gt_seg_maps = dataset.load_annotations()
    metrics, mean_metrics = eval_metrics(results, gt_seg_maps, num_classes=dataset.num_classes, 
                                         ignore_index=dataset.ignore_index, 
                                         reduce_zero_label=dataset.reduce_zeros_label)

    # Removes all_acc from metrics because 
    # it is already included in mean_metrics[0]
    metrics.pop(0)

    return metrics, mean_metrics


"""
    Formato CSV de salida de las métricas:

    Filename,    header1,    header2, ...,    headerN,    mean
    Absolute Acc,all_acc,    -, ...,          -,          all_acc
    Accuracies,  class1_acc, class2_acc, ..., classN_acc, total_mean_acc
    Class Means, class1_iou, class2_iou, ..., classN_iou, total_mIOU
    filename1,   iou_11,     iou_12,...,      iou_1N,     mean_1
    ...
    filenameN,   iou_N1,     iou_12,...,      iou_NN,     mean_N
"""

def format_models_metrics(headers: list, metrics: list, filenames: list, mean_metrics: list, output_file: str):
    """Formats the metrics to csv and outputs to 'output_file'.
    
    Args:
        headers (list[str]): Headers list (e.g. '1','2','3' or 'wall', 'sky', 'earth')
        metrics (list[ndarray]): Metrics(MIoU) break down for every image.
        mean_metrics (list[ndarray]): Mean metrics(MIoU) for the whole batch/dataset.
        output_file (str): File to output the metrics in csv format.
    Returns:
        Nothing.
    """

    ## Takes out total pixel accuracy
    all_acc = mean_metrics[0]

    headers = list(headers)

    headers.insert(0, 'Filename')
    headers.append('Mean')

    filenames = list(filenames)

    filenames.insert(0, 'Class IoUs')
    filenames.insert(0, 'Class Accuracies')
    filenames.insert(0, 'Absolute Accuracy')

    metrics.insert(0, [all_acc] + ['-'] * (len(headers) - 1))

    with open(output_file, 'w', newline='') as outfile:
        csvWriter = csv.DictWriter(outfile, delimiter = ',', 
                                    fieldnames=headers)

        csvWriter.writeheader()
        
        ## Write iou per filename
        for filename, metric, metric_mean in zip(filenames, metrics, mean_metrics):
            info = OrderedDict()
            info[headers[0]] = filename
            for header, iou in zip(headers[1:-1], metric):
                info[header] = iou

            info[headers[-1]] = metric_mean

            csvWriter.writerow(info)
        