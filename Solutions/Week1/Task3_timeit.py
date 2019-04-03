import timeit
import numpy as np

np.random.seed(42)

a = np.random.randint(0, 999999999, 10000).tolist()
x = int(np.random.randint(0, 999999999))
print(x)

b = np.random.randint(0, 100000, 10000).tolist()
y = int(np.random.randint(0, 100000))
print(y)


def func1(list_, _sum):
    """ Task 3a) """
    step = 0
    for elem1 in list_:
        for elem2 in list_:
            step += 1
            if elem1 + elem2 == _sum:
                return True, step
    return False, step


def func2(list_, _sum):
    """ Task 3b) """
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


def func3(list_, _sum):
    """ Task 3c) """
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

    print(func1(a, x))
    print(func2(sorted(a), x))
    print(func3(a, x))
    print(timeit.timeit("func1(a, x)", "from __main__ import func1, a, x", number=3))
    print(timeit.timeit("func2(sorted(a), x)", "from __main__ import func2, a, x", number=3))
    print(timeit.timeit("func3(a, x)", "from __main__ import func3, a, x", number=3))

    print(func1(b, y))
    print(func2(sorted(b), y))
    print(func3(b, y))
    print(timeit.timeit("func1(b, y)", "from __main__ import func1, b, y", number=3))
    print(timeit.timeit("func2(sorted(b), y)", "from __main__ import func2, b, y", number=3))
    print(timeit.timeit("func3(b, y)", "from __main__ import func3, b, y", number=3))
