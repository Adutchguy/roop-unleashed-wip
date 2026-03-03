import threading
import time
import pyvirtualcam


class StreamWriter():
    FPS = 30
    VCam = None
    Active = False
    THREAD_LOCK_STREAM = threading.Lock()
    time_last_process = None
    timespan_min = 0.0
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.Close()

    def __init__(self, size, fps):
        self.time_last_process = time.perf_counter()
        self.FPS = fps
        self.timespan_min = 1.0 / fps
        print('Detecting virtual cam devices')
        self.VCam = pyvirtualcam.Camera(width=size[0], height=size[1], fps=fps, fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False)
        if self.VCam is None:
             print("No virtual camera found!")
             return
        print(f'Using virtual camera: {self.VCam.device}')
        print(f'Using {self.VCam.native_fmt}')
        self.Active = True
         

    def LimitFrames(self):
        while True:
            current_time = time.perf_counter()
            time_passed = current_time - self.time_last_process
            if time_passed >= self.timespan_min:
                break

    # First version used a queue and threading. Surprisingly this
    # totally simple, blocking version is 10 times faster!
    def WriteToStream(self, frame):
        if self.VCam is None:
             return
        with self.THREAD_LOCK_STREAM:
            self.LimitFrames()
            self.VCam.send(frame)
            self.time_last_process = time.perf_counter()
             

    def Close(self):
        self.Active = False
        if self.VCam is None:
            self.VCam.close()
            self.VCam = None




