"""
The get_exp_reg(x, y) function takes in two lists and will do exponential regression on them to return an exponential
function of the form y=ae^(bx) that can be used to make predictions for a given x value.
"""

import math


def get_pearson_correlation_coefficient(x, y):
    """
    Returns correlation coefficient of two lists x, y.

    r = sum((x - mean_x) * (y - mean_y)) / sqrt(sum((x - mean_x)^2) * sum((y - mean_y)^2))
    """
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = 0

    for i in range(len(x)):
        numerator += (x[i] - mean_x) * (y[i] - mean_y)

    sum_squares_x = 0

    for i in range(len(x)):
        sum_squares_x += (x[i] - mean_x)**2

    sum_squares_y = 0

    for i in range(len(y)):
        sum_squares_y += (y[i] - mean_y)**2

    denominator = (sum_squares_x * sum_squares_y)**.5

    return numerator / denominator


def get_sample_std_dev(x):
    """
    Returns sample std dev. x must have a length greater than 1 as the denominator will be length - 1.

    sample_std_dev = sqrt(sum(x - mean_x)^2 / length_x - 1)
    """
    mean_x = sum(x) / len(x)

    sum_squares_x = 0

    for num in x:
        sum_squares_x += (num - mean_x)**2

    return (sum_squares_x / (len(x) - 1))**.5


def get_exp_reg(x, y):
    """
    Exponential regression.

    x and y are lists. x and y must be longer than 1, as length - 1 will be used on denominator for sample std dev.
    Returns function of the form y=ae^(bx).
    To get this, the original formula is manipulated such that linear regression can be used, then it is converted back.

    y=ae^(bx)
    ln(y)=ln(ae^(bx))
    ln(y)=ln(a)+ln(e^(bx))
    ln(y)=ln(a)+bx
    Let z = ln(y). Let c = ln(a).
    z = c + bx

    Linear regression formula being used is:
    b = pearson_correlation_coefficient * sample_std_dev_z / sample_std_dev_x
    c = mean_z - b * mean_x
    """
    # z values = ln(y)
    z_data = []
    for num in y:
        z_data.append(math.log(num, math.e))

    b = get_pearson_correlation_coefficient(x, z_data) * get_sample_std_dev(z_data) / get_sample_std_dev(x)

    mean_z = sum(z_data) / len(z_data)
    mean_x = sum(x) / len(x)

    c = mean_z - (b * mean_x)

    a = math.e**c

    def exp_reg_func(input_value):
        """
        Returns a y value for a given x value using the formula worked out in get_exp_reg().
        """
        print(f'y={a}*e^({b}*x)')
        return a*(math.e**(b*input_value))

    return exp_reg_func


# Example use:
# x = [1, 3, 7, 9, 11, 14]
# y = [19, 16, 13, 12, 11.2, 10.3]
# exp_reg = get_exp_reg(x, y)
# print(exp_reg(20)) -> '7.4235...'
