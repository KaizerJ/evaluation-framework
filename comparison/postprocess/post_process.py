from abc import ABC, abstractmethod

class PostProcess(ABC):

    def __call__(self, image):
        self.process(image)

    @abstractmethod
    def process(self, image):
        pass