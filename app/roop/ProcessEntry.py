class ProcessEntry:
    def __init__(self, filename: str, start: int, end: int, fps: float):
        self.filename = filename
        self.finalname = None
        self.startframe = start
        self.endframe = end
        self.fps = fps