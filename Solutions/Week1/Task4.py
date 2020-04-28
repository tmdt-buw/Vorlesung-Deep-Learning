def func_a(list_, _sum):
    pos1 = 0
    pos2 = len(list_) - 1
    _ans = False
    while not _ans and pos1 != pos2:
        if list_[pos1] + list_[pos2] == _sum:
            _ans = True
        elif list_[pos1] + list_[pos2] < _sum:
            pos1 += 1
        elif list_[pos1] + list_[pos2] > _sum:
            pos2 -= 1
    return _ans


def func_b(list_, _sum):
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
    a = [1, 2, 3, 9]
    b = [1, 2, 4, 4]
    x = 8
    c = [2, 9, 1, 3]
    d = [4, 1, 4, 2]

    """ a) """
    # ans = func_a(a, x)
    # print(ans)
    # ans = func_a(b, x)
    # print(ans)

    """ b) """
    ans = func_b(a, x)
    print(ans)
    ans = func_b(b, x)
    print(ans)
    ans1 = func_a(c, x)
    ans2 = func_b(c, x)
    print(ans1, ans2)
    ans1 = func_a(d, x)
    ans2 = func_b(d, x)
    print(ans1, ans2)
