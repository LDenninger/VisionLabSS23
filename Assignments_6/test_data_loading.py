import time
from contextlib import ContextDecorator
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import numpy as np
import torch
import torchvision.datasets as datasets
from pathlib import Path as P
from run import * 
class Timer(ContextDecorator):
    active_timers = []
    enabled = False
    last_duration = 0.0

    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []

    def __enter__(self):
        if not Timer.enabled:
            return self

        self.start_time = time.time()
        Timer.active_timers.append(self)
        self.children = []
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if not Timer.enabled:
            return

        self.exit_time = time.time()
        self.duration = self.exit_time - self.start_time
        Timer.last_duration = self.duration

        assert Timer.active_timers.pop() == self

        if len(Timer.active_timers) == 0:
            print('Timings:')
            self._print_result()
        else:
            Timer.active_timers[-1]._notify_child(self)

    def _notify_child(self, child):
        self.children.append(child)

    def _print_result(self, indent=0):
        print('{space}{name:{width}}{duration:8.3f}s'.format(
            space=' '*indent,
            name=self.name, width=30-indent,
            duration=self.duration,
        ))
        for child in self.children:
            child._print_result(indent=indent+2)

def smooth_curve(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f

Timer.enabled = True
# Data directory as defined by the PyTorch dataset
DATA_DIR = P(os.getcwd()) / 'data' / 'food101'
NUM_DATA = 1000
NUM_WORKERS = 1
fig_name = 'visLab_env_dataLoader.png'

def test_1():

    dataset = datasets.Food101(root='./data/food101', split='train', download=True)


    ##-- Extract Train Dataset --##
    ds = iter(dataset)
    timings = []

    for i in range(NUM_DATA):
        with Timer("DATA_LOADING"):
            img, label = next(ds)
    
        timings.append(Timer.last_duration)

    return timings

def test_2():
    tts = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64,64))
    ])
    dataset = datasets.Food101(root='./data/food101', split='train', transform=tts, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)

    ##-- Extract Train Dataset --##
    ds = iter(data_loader)
    timings = []

    for i in range(NUM_DATA):
        with Timer("DATA_LOADING"):
            img, label = next(ds)
    
        timings.append(Timer.last_duration)

    return timings

def test_3():

    dataset = Food101Dataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)

    ##-- Extract Train Dataset --##
    ds = iter(data_loader)
    timings = []

    for i in range(NUM_DATA):
        print(f'Iteration {i+1}')
        with Timer("DATA_LOADING"):
            img, label = next(ds)
    
        timings.append(Timer.last_duration)

    return timings


timings = test_3()
plt.style.use('seaborn-v0_8')
plt.plot(smooth_curve(timings, K=31))
plt.xlabel('Iteration')
plt.ylabel('Loading Time in s')
plt.savefig(fig_name)