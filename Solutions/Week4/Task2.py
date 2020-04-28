def val():

    def rel(x):
        return max(0, x)

    x = 2.0
    w1 = 0.25
    w2 = 0.5
    w3 = 0.75
    w4 = 3.3
    b1 = 1.0
    b2 = 0.25
    b3 = 0.25
    b4 = 0.7

    h1 = rel(w1 * x + b1)
    h2 = rel(w2 * h1 + b2)
    h3 = rel(w3 * h2 + b3)
    ystar = w4 * h3 + b4

    print('act', h1, h2, h3)

    return ystar


def loss(ystar):
    return (ystar - 4.0) ** 2


ystar = val()
print('pred', ystar)
l = loss(ystar)
print('loss', l)