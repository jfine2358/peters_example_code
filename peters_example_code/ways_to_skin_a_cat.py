import functools
import random
import math
from functools import partial
from typing import Iterable
import itertools
from numpy import allclose


"""
This document was created to advocate for a new "Reduce-Map" comprehension in python.  
See proposal: https://github.com/petered/peps/blob/master/pep-9999.rst
The idea is to have an initialized generator with a state variable that can be updated in the loop:

  (y := f(y, x) for x in xs from y=initializer)
  
A possible extension would be
  
  (z, y := f(z, x) -> y for x in iter_x from z=initial_z)
  
Which carries state "z" forward but only yields "y" at each iteration. 

Here we use the example of an exponential moving average, which can be described mathematically as:

   y[t] := (1-decay)*y[t-1] + decay*x[t]  : t in [1....], y[0]=0
   
The proposed Python syntax for this operation is:

    smooth_signal = [y := (1-decay)*y + decay*x for x in signal from y=0]

Below, we compare various existing ways of doing this in python:
"""

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
        return self.average

emu = ExponentialMovingAverage(decay=decay)
smooth_signal_3 = [emu(x) for x in signal]
assert allclose(smooth_signal_1, smooth_signal_3)


# 3) Using accumulate ------------------------------------------------
def moving_average_step(average:float, x:float, decay:float):
    return (1-decay)*average + decay*x

smooth_signal_4 = list(itertools.accumulate([0]+list(signal), func=partial(moving_average_step, decay=decay)))[1:]
assert allclose(smooth_signal_1, smooth_signal_4)


# 4) Coroutines (Steven D'Aprano) ---------------------------
# Interesting but confusing for the uninitiated.

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


# 5) A crazy one liner from Serhiy Storchaka: -------------------------------------------
# (nice and succinct but your coworkers might murder you)
smooth_signal_6 = [average for average in [0] for x in signal for average in [(1-decay)*average + decay*x]]
assert allclose(smooth_signal_1, smooth_signal_6)

# OR, if you prefer factored out:

def moving_average_step(average, x, decay):
    return (1-decay)*average + decay*x

smooth_signal_7 = [average for average in [0] for x in signal for average in [moving_average_step(average, x, decay=decay)]]
assert allclose(smooth_signal_1, smooth_signal_7)


# 6) Coroutines take 2 (based on suggestion from Michel Desmoulin)
def exponential_moving_average_couroutine_2(initial=0, decay=decay):
    average = initial
    while True:
        x = (yield average)
        average = (1-decay)*average + decay*x

emac2 = exponential_moving_average_couroutine_2(decay=decay)
emac2.send(None)  # Eaccchhhchchchchchch
smooth_signal_8 = [emac2.send(x) for x in signal]

assert allclose(smooth_signal_1, smooth_signal_8)


# ================= PROPOSED METHODS ==========================
# (comment out below lines to actually run code)


# 7) Peter's proposed syntax:  ------------------------------------------------
smooth_signal_8 = [average := (1-decay)*average + decay*x for x in signal from average=0]

# OR, if you prefer factored out:
def moving_average_step(average, x, decay):
    return (1-decay)*average + decay*x

smooth_signal_7 = [average = moving_average_step(average, x, decay=decay) for x in signal from average=0]


# 8) Proposals based on statement-local-name-bindings :  ------------------------------------------------

def moving_average_step(average, x, decay):
    return (1-decay)*average + decay*x

average=0
[(moving_average_step(average, x, decay=decay) as average) for x in signal]

average=0
[(average := moving_average_step(average, x, decay=decay)) for x in signal]
