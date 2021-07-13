# Semantic Segmentation Comparison

This repository contains the codebase for inferring and evaluating different semantic segmentation models which follow the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) scheme.
This code was developed as part of the undergraduate thesis "Analysis of coastal images in turistic environments" presented at ULPGC.

## HOWTO

1. Install MMSegmentation repository and its dependecies.
2. Write the [config file](./config/config.json):
    
    Expected JSON config:
    ```json
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
    ```
    With the original images folder and the annotations as grey images with values between 0 and 150. <br>
    `models` should be a list with routes to the MMSegmentation-like config and checkpoints.

3. Run either `compare_pipeline.py` for default or `compare_postprocess_pipeline.py` to execute and compute the results.

## TODO
- Adding postproccessings: 
    1. Think about others

## Authors
- Jonay Suárez Ramírez