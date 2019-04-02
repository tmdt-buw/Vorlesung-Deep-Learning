a = [1, 2, 3, 9]
b = [1, 2, 4, 4]
x = 8
c = [2, 9, 1, 3]
d = [4, 1, 4, 2]


def func1(list_, _sum):
    """ Task 3a) """
    for elem1 in list_:
        for elem2 in list_:
            if elem1 + elem2 == _sum:
                return True
    return False


def func2(list_, _sum):
    """ Task 3b) """
    pos1 = 0
    pos2 = len(list_) - 1
    _ans = False
    while not _ans and pos1 != pos2:
        if list_[pos1] + list_[pos2] == _sum:
            _ans = True
        elif list_[pos1] + list_[pos2] < x:
            pos1 += 1
        elif list_[pos1] + list_[pos2] > x:
            pos2 -= 1
    return _ans


def func3(list_, _sum):
    """ Task 3c) """
    pos = 0
    _ans = False
    store = set()
    while not _ans and pos < len(list_):
        if _sum - list_[pos] in store:
            _ans = True
        else:
            store.add(list_[pos])
            pos += 1
    return _ans


if __name__ == "__main__":

    """ Task 3a) """
    ans = func1(a, x)
    print(ans)
    ans = func1(b, x)
    print(ans)

    """ Task 3b) """
    ans = func2(a, x)
    print(ans)
    ans = func2(b, x)
    print(ans)

    """ Task 3c) """
    ans = func3(a, x)
    print(ans)
    ans = func3(b, x)
    print(ans)
    ans1 = func2(c, x)
    ans2 = func3(c, x)
    print(ans1, ans2)
    ans1 = func2(d, x)
    ans2 = func3(d, x)
    print(ans1, ans2)
