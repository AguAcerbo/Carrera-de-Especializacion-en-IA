import numpy as np


def gradient_descent(x_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n = x_train.shape[0]
    m = x_train.shape[1]

    # initialize random weights
    w = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        prediction = np.matmul(x_train, w)                      # nx1
        error = y_train - prediction                            # nx1

        grad_sum = np.sum(error * x_train, axis=0)
        grad_mul = -2/n * grad_sum                              # 1xm
        gradient = np.transpose(grad_mul).reshape(-1, 1)        # mx1

        w = w - (lr * gradient)

    return w


def stochastic_gradient_descent(x_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        w = mx1
    """
    n = x_train.shape[0]
    m = x_train.shape[1]

    # initialize random weights
    w = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]

        for j in range(n):
            prediction = np.matmul(x_train[j].reshape(1, -1), w)    # 1x1
            error = y_train[j] - prediction                         # 1x1

            grad_sum = error * x_train[j]
            grad_mul = -2/n * grad_sum                              # 2x1
            gradient = np.transpose(grad_mul).reshape(-1, 1)        # 2x1

            w = w - (lr * gradient)

    return w


def mini_batch_gradient_descent(x_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
    n = x_train.shape[0]
    m = x_train.shape[1]

    # initialize random weights
    w = np.random.randn(m).reshape(m, 1)

    for epoch in range(amt_epochs):
        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(x_train) / b)
        for batch in range(0, len(x_train), batch_size):
            end = batch + batch_size if batch + batch_size <= len(x_train) else len(x_train)
            batch_x = x_train[batch: end]
            batch_y = y_train[batch: end]

            prediction = np.matmul(batch_x, w)                  # nx1
            error = batch_y - prediction                        # nx1

            grad_sum = np.sum(error * batch_x, axis=0)
            grad_mul = -2/n * grad_sum                          # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)    # mx1

            w = w - (lr * gradient)

    return w
