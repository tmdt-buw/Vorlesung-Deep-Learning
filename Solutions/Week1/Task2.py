def f_faku(n):
    faku = 1
    for i in range(1, n + 1):
        faku *= i
    return faku


def f_faku_rec(n):
    if n <= 1:
        faku = 1
    else:
        faku = n * (f_faku_rec(n - 1))
    return faku


if __name__ == "__main__":
    faku = f_faku(5)
    print(faku)

    faku = f_faku_rec(5)
    print(faku)