
class PostProcessPipeline:
    
    def __init__(self, post_processes = ()):
        self.post_processes = list(post_processes)

    def __call__(self, image):
        for post_process in self.post_processes:
            post_process(image)