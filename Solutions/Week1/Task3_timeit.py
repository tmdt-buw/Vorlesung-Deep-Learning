import timeit
import numpy as np


def func1():
    list_ = list(np.random.randint(0, 9999, 1000))
    _sum = int(np.random.randint(0, 9999))
    step = 0
    for elem1 in list_:
        for elem2 in list_:
            step += 1
            if elem1 + elem2 == _sum:
                return True, step
    return False, step


def func2():
    list_ = sorted(list(np.random.randint(0, 9999, 1000)))
    _sum = int(np.random.randint(0, 9999))
    pos1 = 0
    pos2 = len(list_) - 1
    _ans = False
    step = 0
    while not _ans and pos1 <= pos2:
        step += 1
        step_sum = list_[pos1] + list_[pos2]
        if step_sum == _sum:
            _ans = True
        elif step_sum < _sum:
            pos1 += 1
        else:
            pos2 -= 1
    return _ans, step


def func3():
    list_ = list(np.random.randint(0, 9999, 1000))
    _sum = int(np.random.randint(0, 9999))
    pos = 0
    _ans = False
    store = set()
    step = 0
    while not _ans and pos < len(list_):
        step += 1
        if _sum - list_[pos] in store:
            _ans = True
        else:
            store.add(list_[pos])
            pos += 1
    return _ans, step


if __name__ == "__main__":

    np.random.seed(42)
    print(timeit.timeit("func1()", "from __main__ import func1", number=1000))
    np.random.seed(42)
    print(timeit.timeit("func2()", "from __main__ import func2", number=1000))
    np.random.seed(42)
    print(timeit.timeit("func3()", "from __main__ import func3", number=1000))
