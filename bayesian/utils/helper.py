import numpy as np
from scipy.stats import norm


def confidence_interval(num_samples, p, alpha=0.95):

    if not 0 <= p <= 1:
        raise ValueError("The estimated probability 'p_hat' must be between 0 and 1.")
    
    if not 0 < alpha < 1:
        raise ValueError("The 'confidence_level' must be between 0 and 1.")

    if num_samples == 0:
        return (0, 1)

    z_score = norm.ppf(1 - (1 - alpha) / 2)
    std_error = np.sqrt((p * (1 - p)) / num_samples)

    margin_of_error = z_score * std_error

    lower_bound = p - margin_of_error
    upper_bound = p + margin_of_error

    lower_bound = max(0, lower_bound)
    upper_bound = min(1, upper_bound)

    return (lower_bound, upper_bound)

def process_logs(log_path):

    res = {}

    with open(log_path, 'r') as f:
        lines = f.readlines()

        index = 0
        for i, line in enumerate(lines):
            if "size" in line:
                msg = line.split(" - INFO - ")[1]
                key = msg.split(" size: ")[0].strip()
                index = i
            elif i-index == 2:
                value = line.split(": ")[0].strip()
                value = value.split()[-1]
                res[key] = value
    return res

def room_conversion(x):
    if x <= 4:
        return "1-4"
    else:
        return "5"