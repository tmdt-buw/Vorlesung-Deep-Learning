def f_fibo(n):
    fibo = [0, 1]
    for i in range(n - 1):
        fibo.append(fibo[-1] + fibo[-2])
    return fibo[1:]


def f_fibo_rec(n):
    if n <= 1:
        return n
    else:
        return f_fibo_rec(n - 1) + f_fibo_rec(n - 2)


if __name__ == "__main__":
    fibos = f_fibo(10)
    print(fibos)

    fibos = f_fibo_rec(10)
    print(fibos)