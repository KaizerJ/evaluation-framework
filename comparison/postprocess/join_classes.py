from comparison.postprocess.post_process import PostProcess
import numpy

class JoinClasses(PostProcess):

    def __init__(self, classes, joint_class):
        self.classes = classes
        self.joint_class = joint_class

    def process(self, image):
        
        for c in self.classes:
            image[ image == c ] = self.joint_class
