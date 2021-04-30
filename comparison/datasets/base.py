import os
import cv2

def list_full_filenames(dirName):
    full_filenames = list()
    for filename in os.listdir(dirName):
        full_filenames.append(os.path.join(dirName, filename))

    return full_filenames

class BaseDataset():
    
    def __init__(self,
                 img_dir,
                 annotations_dir,
                 num_classes,
                 img_suffix,
                 ann_suffix,
                 ignore_index=255,
                 reduce_zero_label=True):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.num_classes = num_classes
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.ignore_index = ignore_index
        self.reduce_zeros_label = reduce_zero_label

        self.images_filenames = list_full_filenames(self.img_dir)
        self.ann_filenames = list_full_filenames(self.annotations_dir)

        if len(self.images_filenames) != len(self.ann_filenames):
            raise Exception('Number of images and annotations is not the same')

        self.images = None
        self.annotations = None

    
    def load_images(self):

        if self.images == None:
            self.images = [cv2.imread(filename, cv2.IMREAD_COLOR) for filename in self.images_filenames]

        return self.images

    def load_annotations(self):

        if self.annotations == None:
            self.annotations = [cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in self.ann_filenames]
        
        return self.annotations