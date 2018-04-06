import functools
import random
import math
from functools import partial
from typing import Iterable
import itertools
from numpy import allclose

signal = [math.sin(i*0.01) + random.normalvariate(0, 0.1) for i in range(1000)]
decay = 0.1

# ================= EXISTING METHODS ==========================


# 1) The orthodox way ------------------------------------------------
def exponential_moving_average_1(signal: Iterable[float], decay: float, initial_value: float=0.):
    running_average = []
    average = initial_value
    for xt in signal:
        average = (1-decay)*average + decay*xt
        running_average.append(average)
    return running_average


smooth_signal_1 = exponential_moving_average_1(signal, decay=decay)

# 2) A more python3ic way ------------------------------------------------
def exponential_moving_average_2(signal: Iterable[float], decay: float, initial_value: float=0.):
    average = initial_value
    for xt in signal:
        average = (1-decay)*average + decay*xt
        yield average

smooth_signal_2 = list(exponential_moving_average_2(signal, decay=decay))
assert allclose(smooth_signal_1, smooth_signal_2)


# 3) Using a class ------------------------------------------------

class ExponentialMovingAverage(object):

    def __init__(self, decay, initial_value=0):
        self.decay=decay
        self.average = initial_value

    def __call__(self, x):
        self.average = (1-decay)*self.average + decay*x

emu = ExponentialMovingAverage(decay=decay)
smooth_signal_3 = list(exponential_moving_average_2(signal, decay=decay))
assert allclose(smooth_signal_1, smooth_signal_3)


# 3) Using accumulate ------------------------------------------------
def moving_average_step(average:float, x:float, decay:float):
    return (1-decay)*average + decay*x

smooth_signal_4 = list(itertools.accumulate([0]+list(signal), func=partial(moving_average_step, decay=decay)))[1:]
assert allclose(smooth_signal_1, smooth_signal_4)


# 4) Coroutines ------------------------------------------------
def coroutine(func):
    """Decorator to prime coroutines when they are initialised."""
    @functools.wraps(func)
    def started(*args, **kwargs):
        cr = func(*args,**kwargs)
        cr.send(None)
        return cr
    return started


@coroutine
def exponential_moving_average_coroutine(decay: float, initial_value: float=0.):

    average = initial_value
    x = (yield average)
    while True:
        average = decay*x + (1-decay)*average
        x = (yield average)


aver = exponential_moving_average_coroutine(decay=decay)
smooth_signal_5 = [aver.send(x) for x in signal]
assert allclose(smooth_signal_1, smooth_signal_5)


# 5) A crazy one liner from Serhiy: -------------------------------------------
# (nice and succinct but your coworkers might murder you)
smooth_signal_6 = [average for average in [0] for x in signal for average in [(1-decay)*average + decay*x]]
assert allclose(smooth_signal_1, smooth_signal_6)

# OR, if you prefer factored out:

def moving_average_step(average, x, decay):
    return (1-decay)*average + decay*x

smooth_signal_7 = [average for average in [0] for x in signal for average in [moving_average_step(average, x, decay=decay)]]
assert allclose(smooth_signal_1, smooth_signal_7)


# ================= PROPOSED METHODS ==========================
# (comment out below lines to actually run code)


# 6) Peter's proposed syntax:  ------------------------------------------------
smooth_signal_8 = [average = (1-decay)*average + decay*x for x in signal from average=0]

# OR, if you prefer factored out:
def moving_average_step(average, x, decay):
    return (1-decay)*average + decay*x

smooth_signal_7 = [average = moving_average_step(average, x, decay=decay) for x in signal from average=0]


# 6) Proposals based on statement-local-name-bindings :  ------------------------------------------------

def moving_average_step(average, x, decay):
    return (1-decay)*average + decay*x

average=0
[(moving_average_step(average, x, decay=decay) as average) for x in signal]

average=0
[(average := moving_average_step(average, x, decay=decay)) for x in signal]
