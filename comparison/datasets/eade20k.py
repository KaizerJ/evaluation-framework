from comparison.datasets.base import BaseDataset
from comparison.datasets.ade20k import ade20k

class ExtendedAde20k(BaseDataset):

    """Extended ADE20K dataset.
    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories, extended or virtual classes are whose index are after 150. 
    ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to '.png'.
    """

    def __init__(self, img_dir,
                       ann_dir,
                       ann_post_process_pipeline,
                       virtual_classes=(),
                       virtual_palette=()):
        super().__init__(img_dir,
                         ann_dir,
                         num_classes=150 + len(virtual_classes),
                         img_suffix='.jpg',
                         ann_suffix='.png',
                         ann_postprocess_pipeline=ann_post_process_pipeline)

        self.CLASSES = ade20k.CLASSES + virtual_classes
        self.PALETTE = ade20k.PALETTE + virtual_palette
