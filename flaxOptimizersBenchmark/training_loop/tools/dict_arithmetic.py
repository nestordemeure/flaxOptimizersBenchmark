import copy
import numbers
import math

#----------------------------------------------------------------------------------------
# HIGHER ORDER FUNCTIONS

def map_dict(d, func):
    """
    applies func to all the numbers in dict
    """
    for (name,value) in d.items():
        if isinstance(value, numbers.Number):
            d[name] = func(value)
        elif isinstance(value, list):
            for i in len(value): value[i] = func(value[i])
        elif isinstance(value, dict):
            map_dict(value, func)

#----------------------------------------------------------------------------------------
# UNARY OPERATIONS

def zero_dict(d):
    """
    creates a copy of d that is filled with zeroes
    """
    def to_zero(x): return 0 if isinstance(x, int) else 0.0
    result = copy.deepcopy(d)
    map_dict(result, to_zero)
    return result

def divide_dict_scalar(d, denominator):
    """
    divides all the elements in a dictionary by a given denominator
    """
    map_dict(d, lambda x: x/denominator)

def sqrt_dict(d):
    """
    divides all the elements in a dictionary by a given denominator
    """
    map_dict(d, math.sqrt)

#----------------------------------------------------------------------------------------
# MULTIARY OPERATIONS

def add_dict(d_out, d_in):
    """
    adds d_in to d_out
    the sum is only applied to the numbers and list of numbers
    """
    for (name,value_out) in d_out.items():
        if isinstance(value_out, numbers.Number):
            d_out[name] += d_in[name]
        elif isinstance(value_out, list):
            value_in = d_in[name]
            for i in len(value_out): value_out[i] += value_in[i]
        elif isinstance(value_out, dict):
            add_dict(value_out, d_in[name])

def add_diff_square(d_out, d1, d2):
    """
    computes d_out += (d1-d2)²
    the operation is only applied to the numbers and list of numbers
    """
    for (name,value_out) in d_out.items():
        if isinstance(value_out, numbers.Number):
            d_out[name] += (d1[name] - d2[name])**2
        elif isinstance(value_out, list):
            value1 = d1[name]
            value2 = d2[name]
            for i in len(value_out):
                value_out[i] += (value1[i] - value2[i])**2
        elif isinstance(value_out, dict):
            add_diff_square(value_out, d1[name], d2[name])

#----------------------------------------------------------------------------------------
# STATISTICS

def average_dic(dict_list):
    """
    computes the average of a list of dictionaries
    the average is only applied to the numbers and list of numbers
    """
    result = copy.deepcopy(dict_list[0])
    for d in dict_list[1:]:
        add_dict(result, d)
    divide_dict_scalar(result, len(dict_list))
    return result

def variance_dict(dict_list, average, unbiasing=-1):
    """
    computes the variance of a list of dictionary, given their average
    """
    result = zero_dict(average)
    for d in dict_list:
        add_diff_square(result, d, average)
    divide_dict_scalar(result, len(dict_list)+unbiasing)
    return result

def standard_deviation_dict(dict_list, average):
    """
    computes the standard deviation of a list of dictionary, given their average
    """
    result = variance_dict(dict_list, average, unbiasing=-1.5)
    sqrt_dict(result)
    return result
