"""
Tools for generating task sets.
"""
from functools import reduce 
import numpy as np
import random
import math
import itertools


def UUniFastDiscard(n, u, nsets):
    sets = []
    while len(sets) < nsets:
        # Classic UUniFast algorithm:
        utilizations = []
        sumU = u
        for i in range(1, n):
            nextSumU = sumU * random.random() ** (1.0 / (n - i))
            utilizations.append(sumU - nextSumU)
            sumU = nextSumU
        utilizations.append(nextSumU)

        # If no task utilization exceeds 1:
        if not [ut for ut in utilizations if ut > 1]:
            sets.append(utilizations)

    return sets


def StaffordRandFixedSum(n, u, nsets):
    """
    Copyright 2010 Paul Emberson, Roger Stafford, Robert Davis.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
    EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
    OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    The views and conclusions contained in the software and documentation are
    those of the authors and should not be interpreted as representing official
    policies, either expressed or implied, of Paul Emberson, Roger Stafford or
    Robert Davis.

    Includes Python implementation of Roger Stafford's randfixedsum implementation
    http://www.mathworks.com/matlabcentral/fileexchange/9700
    Adapted specifically for the purpose of taskset generation with fixed
    total utilisation value

    Please contact paule@rapitasystems.com or robdavis@cs.york.ac.uk if you have
    any questions regarding this software.
    """
    if n < u:
        print "n < u ", n,u
        return None

    #deal with n=1 case
    if n == 1:
        return np.tile(np.array([u]), [nsets, 1])

    k = min(int(u), n - 1)
    s = u
    s1 = s - np.arange(k, k - n, -1.)
    s2 = np.arange(k + n, k, -1.) - s

    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max

    w = np.zeros((n, n + 1))
    w[0, 1] = huge
    t = np.zeros((n - 1, n))

    for i in np.arange(2, n + 1):
        tmp1 = w[i - 2, np.arange(1, i + 1)] * s1[np.arange(0, i)] / float(i)
        tmp2 = w[i - 2, np.arange(0, i)] * s2[np.arange(n - i, n)] / float(i)
        w[i - 1, np.arange(1, i + 1)] = tmp1 + tmp2
        tmp3 = w[i - 1, np.arange(1, i + 1)] + tiny
        tmp4 = s2[np.arange(n - i, n)] > s1[np.arange(0, i)]
        t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + \
            (1 - tmp1 / tmp3) * (np.logical_not(tmp4))

    x = np.zeros((n, nsets))
    rt = np.random.uniform(size=(n - 1, nsets))  # rand simplex type
    rs = np.random.uniform(size=(n - 1, nsets))  # rand position in simplex
    s = np.repeat(s, nsets)
    j = np.repeat(k + 1, nsets)
    sm = np.repeat(0, nsets)
    pr = np.repeat(1, nsets)

    for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
        # decide which direction to move in this dimension (1 or 0):
        e = rt[(n - i) - 1, ...] <= t[i - 1, j - 1]
        sx = rs[(n - i) - 1, ...] ** (1.0 / i)  # next simplex coord
        sm = sm + (1.0 - sx) * pr * s / (i + 1)
        pr = sx * pr
        x[(n - i) - 1, ...] = sm + pr * e
        s = s - e
        j = j - e  # change transition table column if required

    x[n - 1, ...] = sm + pr * s

    #iterated in fixed dimension order but needs to be randomised
    #permute x row order within each column
    for i in range(0, nsets):
        x[..., i] = x[np.random.permutation(n), i]

    return x.T.tolist()


def gen_ripoll(nsets, compute, deadline, period, target_util):
    """
    Ripoll et al. tasksets generator.

    Args:
        - `nsets`: Number of tasksets to generate.
        - `compute`: Maximum computation time of a task.
        - `deadline`: Maximum slack time.
        - `period`: Maximum delay after the deadline.
        - `target_util`: Total utilization to reach.
    """
    sets = []
    for i in range(nsets):
        task_set = []
        total_util = 0.0
        while total_util < target_util:
            c = random.randint(1, compute)
            d = c + random.randint(0, deadline)
            p = d + random.randint(0, period)
            task_set.append((c, d, p))
            total_util += c / p
        sets.append(task_set)
    return sets


def gen_uunifastdiscard(nsets, u, n):
    """
    The UUniFast algorithm was proposed by Bini for generating task
    utilizations on uniprocessor architectures.

    The UUniFast-Discard algorithm extends it to multiprocessor by
    discarding task sets containing any utilization that exceeds 1.

    This algorithm is easy and widely used. However, it suffers from very
    long computation times when n is close to u. Stafford's algorithm is
    faster.

    Args:
        - `n`: The number of tasks in a task set.
        - `u`: Total utilization of the task set.
        - `nsets`: Number of sets to generate.

    Returns `nsets` of `n` task utilizations.
    """
    return UUniFastDiscard(u, n, nsets)


def gen_randfixedsum(nsets, u, n):
    """
    Stafford's RandFixedSum algorithm implementated in Python.

    Based on the Python implementation given by Paul Emberson, Roger Stafford,
    and Robert Davis. Available under the Simplified BSD License.

    Args:
        - `n`: The number of tasks in a task set.
        - `u`: Total utilization of the task set.
        - `nsets`: Number of sets to generate.
    """
    return StaffordRandFixedSum(u, n, nsets)


def gen_kato_utilizations(nsets, umin, umax, target_util):
    """
    Kato et al. tasksets generator.

    Args:
        - `nsets`: Number of tasksets to generate.
        - `umin`: Minimum task utilization.
        - `umax`: Maximum task utilization.
        - `target_util`:
    """
    sets = []
    for i in range(nsets):
        task_set = []
        total_util = 0.0
        while total_util < target_util:
            u = random.uniform(umin, umax)
            if u + total_util > target_util:
                u = target_util - total_util
            total_util += u
            task_set.append(u)
        sets.append(task_set)
    return sets


def next_arrival_poisson(period):
    return -math.log(1.0 - random.random()) * period


def gen_arrivals(period, min_, max_, round_to_int=False):
    def trunc(x, p):
        return int(x * 10 ** p) / float(10 ** p)

    dates = []
    n = min_ - period
    while True:
        n += next_arrival_poisson(period) + period
        if round_to_int:
            n = int(round(n))
        else:
            n = trunc(n, 6)
        if n > max_:
            break
        dates.append(n)
    return dates


def gen_periods_loguniform(n, nsets, min_, max_, round_to_int=False):
    """
    Generate a list of `nsets` sets containing each `n` random periods using a
    loguniform distribution.

    Args:
        - `n`: The number of tasks in a task set.
        - `nsets`: Number of sets to generate.
        - `min_`: Period min.
        - `max_`: Period max.
    """
    periods = np.exp(np.random.uniform(low=np.log(min_), high=np.log(max_),
                                       size=(nsets, n)))
    if round_to_int:
        return np.rint(periods).tolist()
    else:
        return periods.tolist()

def lcm(a, b):
    if a > b:
        greater = a
    else:
        greater = b

    while True:
        if greater % a == 0 and greater % b == 0:
            lcm = greater
            break
        greater += 1

    return lcm

def get_lcm(values):
    return reduce(lambda x, y: lcm(x, y), values)

def gen_periods_hyperperiod(n, nsets, min_, max_, hyperperiod, round_to_int=False):

    p_list = []

    while len(p_list) != nsets:

        periods = np.random.randint(low=1, high=11, size=(nsets, n))
        periods = [i*10 for i in periods]

        for i in periods:
            if get_lcm(i) <= hyperperiod:
                p_list.append(i)
                
                if len(p_list) == nsets:
                    return p_list


    return p_list

def gen_int_periods_uniform_hype(n, nsets, min_, max_, step, hyperperiod, group_by = 0):
    """
    Generate a list of `nsets` sets containing each `n` random periods using a
    uniform distribution.

    Args:
        - `n`: The number of tasks in a task set.
        - `nsets`: Number of sets to generate.
        - `min_`: Period min.
        - `max_`: Period max.
        - `hyperperiod`: Hyper-period max.
    """
    p_list = []

    if group_by > 0:
        n = n/group_by

    while len(p_list) != nsets:
        periods = [] #np.random.randint(low=min_/step, high=(max_/step)+1, size=(nsets, n))
        
        for i in range(0, nsets):
            periods.append([random.randrange(min_, max_+1, step) for _ in range(n)])

        for i in range(0, len(periods)):
            if get_lcm(periods[i]) <= hyperperiod:
                print get_lcm(periods[i])
                if group_by > 0:
                    found = 0
                    list_ = periods[i]
                    for y in range(i+1, len(periods)):
                        if get_lcm(periods[y]) <= hyperperiod:
                            found += 1
                            list_.extend(periods[y])

                            if found == group_by - 1:
                                p_list.append(list_)
                                break
                else:
                    p_list.append(periods[i])
                #print len(p_list)
                
                if len(p_list) == nsets:
                    return p_list
    return p_list

def gen_int_periods_uniform_hype_pow2(n, nsets, min_, max_, hyperperiod):
    """
    Generate a list of `nsets` sets containing each `n` random periods using a
    uniform distribution.

    Args:
        - `n`: The number of tasks in a task set.
        - `nsets`: Number of sets to generate.
        - `min_`: Period min.
        - `max_`: Period max.
        - `hyperperiod`: Hyper-period max.
    """
    p_list = []

    while len(p_list) != nsets:
        periods = np.random.randint(low=1, high=10, size=(nsets, n))
        periods = [2**i for i in periods]

        for i in periods:
            if get_lcm(i) <= hyperperiod:
                #print i
                print get_lcm(i)
                p_list.append(i)
                
                if len(p_list) == nsets:
                    return p_list
    return p_list

def gen_periods_harmonic(n, nsets, min_, max_):

    periods = np.random.randint(low=1, high=10, size=(nsets, n))

    periods = [2**i for i in periods]

    #print periods


    return periods



def gen_periods_uniform(n, nsets, min_, max_, round_to_int=False):
    """
    Generate a list of `nsets` sets containing each `n` random periods using a
    uniform distribution.

    Args:
        - `n`: The number of tasks in a task set.
        - `nsets`: Number of sets to generate.
        - `min_`: Period min.
        - `max_`: Period max.
    """
    periods = np.random.uniform(low=min_, high=max_, size=(nsets, n))

    if round_to_int:
        return np.rint(periods).tolist()
    else:
        return periods.tolist()

def gen_periods_(n, nsets, p_list):
    """
    Generate a list of `nsets` sets containing each `n` random periods using a
    uniform distribution.

    Args:
        - `n`: The number of tasks in a task set.
        - `nsets`: Number of sets to generate.
    """

    periods = []
    x = 0

    #for x in range(0, nsets):
    while len(periods) != nsets:
        set_ = []
        index = np.random.randint(low=0, high=len(p_list), size=n)
        for i in index:
            set_.append(p_list[i])

        if get_lcm(set_) <= 1000:
            periods.append(set_)

    return periods

def gen_periods_hype100(n, nsets):
    possibilities = [#[10], [20], [30], [40], [50], [60],[70], [80], [90], [100], 
    [10, 20], [10, 30], 
     [10, 40], [10, 50], [10, 60], [10, 70], [10, 80], 
     [10, 90], [10, 100], [20, 30], [20, 40], [20, 50], 
     [20, 60], [20, 80], [20, 100], [30, 60], [30, 90], 
     [40, 80], [50, 100], [10, 20, 30], [10, 20, 40], 
     [10, 20, 50], [10, 20, 60], [10, 20, 80], 
     [10, 20, 100], [10, 30, 60], [10, 30, 90], 
     [10, 40, 80], [10, 50, 100], [20, 30, 60], 
     [20, 40, 80], [20, 50, 100], [10, 20, 30, 60], 
     [10, 20, 40, 80], [10, 20, 50, 100]]


    index = np.random.randint(low=0, high=len(possibilities), size=nsets)
    periods = []

    for i in index:
        # fill whit the sames values on i index
        set_ = possibilities[i] * 10

        #store n periods per set
        periods.append(set_[0:n])

    return periods

def gen_periods_hype250(n, nsets):
    possibilities = [#[25], [50], [75], [100], [125], [150], [175], [200], [225], [250], 
    [25, 50], [25, 75], [25, 100], [25, 125], [25, 150], [25, 175], 
    [25, 200], [25, 225], [25, 250], [50, 75], [50, 100], [50, 125], 
    [50, 150], [50, 200], [50, 250], [75, 150], [75, 225], [100, 200],
    [125, 250], [25, 50, 75], [25, 50, 100], [25, 50, 125], 
    [25, 50, 150], [25, 50, 200], [25, 50, 250], [25, 75, 150], 
    [25, 75, 225], [25, 100, 200], [25, 125, 250], [50, 75, 150], 
    [50, 100, 200], [50, 125, 250], [25, 50, 75, 150], 
    [25, 50, 100, 200], [25, 50, 125, 250]]


    index = np.random.randint(low=0, high=len(possibilities), size=nsets)
    periods = []

    for i in index:
        # fill whit the sames values on i index
        set_ = possibilities[i] * 10

        #store n periods per set
        periods.append(set_[0:n])

    return periods

def gen_periods_hype80(n, nsets):
    possibilities = [#[8], [16], [24], [32], [40], [48], [56], [64], [72], [80], 
    [8, 16], [8, 24], [8, 32], [8, 40], [8, 48], [8, 56], [8, 64], 
    [8, 72], [8, 80], [16, 24], [16, 32], [16, 40], [16, 48], 
    [16, 64], [16, 80], [24, 48], [24, 72], [32, 64], [40, 80], 
    [8, 16, 24], [8, 16, 32], [8, 16, 40], [8, 16, 48], [8, 16, 64], 
    [8, 16, 80], [8, 24, 48], [8, 24, 72], [8, 32, 64], [8, 40, 80], 
    [16, 24, 48], [16, 32, 64], [16, 40, 80], [8, 16, 24, 48], 
    [8, 16, 32, 64], [8, 16, 40, 80]]


    index = np.random.randint(low=0, high=len(possibilities), size=nsets)
    periods = []

    for i in index:
        # fill whit the sames values on i index
        set_ = possibilities[i] * 10

        #store n periods per set
        periods.append(set_[0:n])

    return periods


def gen_set_periods_hype(n, nsets, p_list, hype):

    set_ = []

    for L in range(1, len(p_list)+1):
        for subset in itertools.combinations(p_list, L):
            if get_lcm(list(subset)) <= hype:
                if list(subset) not in set_:
                    set_.append(list(subset))
    
    print set_

# gera conjuntos com hyperperiodo limitado
def get_set_periods(n, nsets, p_list, hype):

    periods = []

    while len(periods) < nsets:
        set_ = []
        while len(set_) < n:
            index = np.random.randint(low=0, high=len(p_list), size=nsets)
            for i in index:
                if len(set_) == 0:
                    set_.append(p_list[i])
                elif len(set_) < n:
                    if get_lcm(list(set_+[p_list[i]])) <= hype:
                        set_.append(p_list[i])
        periods.append(set_)

    return periods

def gen_set_by_list(n, nsets, p_list):

    periods = []

    while len(periods) < nsets:
        set_ = p_list * n
        #store n periods per set
        periods.append(set_[0:n])

    return periods


def gen_periods_discrete(n, nsets, periods):
    """
    Generate a matrix of (nsets x n) random periods chosen randomly in the
    list of periods.

    Args:
        - `n`: The number of tasks in a task set.
        - `nsets`: Number of sets to generate.
        - `periods`: A list of available periods.
    """
    try:
        return np.random.choice(periods, size=(nsets, n)).tolist()
    except AttributeError:
        # Numpy < 1.7:
        p = np.array(periods)
        return p[np.random.randint(len(p), size=(nsets, n))].tolist()


def gen_tasksets(utilizations, periods):
    """
    Take a list of task utilization sets and a list of task period sets and
    return a list of couples (c, p) sets. The computation times are truncated
    at a precision of 10^-10 to avoid floating point precision errors.

    Args:
        - `utilization`: The list of task utilization sets. For example::

            [[0.3, 0.4, 0.8], [0.1, 0.9, 0.5]]
        - `periods`: The list of task period sets. For examples::

            [[100, 50, 1000], [200, 500, 10]]

    Returns:
        For the above example, it returns::

            [[(30.0, 100), (20.0, 50), (800.0, 1000)],
             [(20.0, 200), (450.0, 500), (5.0, 10)]]
    """
    def trunc(x, p):
        return int(x * 10 ** p) / float(10 ** p)

    return [[(trunc(ui * pi, 6), trunc(pi, 6)) for ui, pi in zip(us, ps)]
            for us, ps in zip(utilizations, periods)]
