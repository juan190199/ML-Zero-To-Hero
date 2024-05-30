def soft_thresholding_operator(self, x, lambda_):
    if x > 0 and lambda_ < abs(x):
        return x - lambda_
    elif x < 0 and lambda_ < abs(x):
        return x + lambda_
    else:
        return 0.0