def func1(list_, _sum):
    for elem1 in list_:
        for elem2 in list_:
            if elem1 + elem2 == _sum:
                return True
    return False


if __name__ == "__main__":
    a = [1, 2, 3, 9]
    b = [1, 2, 4, 4]
    x = 8

    """ Task 3a) """
    ans = func1(a, x)
    print(ans)
    ans = func1(b, x)
    print(ans)

    x =