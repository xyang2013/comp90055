import re
import sklearn.utils
import numpy as np

def transform(input_file, output_file, regular_exp):
    """Transform input_file to output_file containing only lines that match regular_exp."""
    lines = []
    with open(input_file) as ifile:
        lines.append(ifile.readline()) # append header line
        for line in ifile:
            if re.match(regular_exp, line):
                lines.append(line)
    with open(output_file, 'w') as ofile:
        for line in lines:
            ofile.write(line)

def convert_to_timestamp(x, helper, tz):
    """Convert string to  timestamp with timezone.

    Keyword arguments:
        x -- the string containing timestamp information
        helper -- the helper function that converts x to timestamp object
        tz -- the designated timezone
    """
    dt = helper(x)
    tz_dt = dt.astimezone(tz)
    return tz_dt

def fit_sample(X, y, seed):
    """Resample the dataset such that the proportions of all labels are equal.
    """
    label = np.unique(y)
    stats_c_ = {}
    maj_n = 0
    for i in label:
        nk = sum(y==i)
        stats_c_[i] = nk
        if nk > maj_n:
            maj_n = nk
            maj_c_ = i

    # Keep the samples from the majority class
    X_resampled = X[y == maj_c_]
    y_resampled = y[y == maj_c_]
    # Loop over the other classes over picking at random
    for key in stats_c_.keys():

        # If this is the majority class, skip it
        if key == maj_c_:
            continue

        # Define the number of sample to create
        num_samples = int(stats_c_[maj_c_] -stats_c_[key])

        # Pick some elements at random
        random_state = sklearn.utils.check_random_state(seed)
        indx = random_state.randint(low=0, high=stats_c_[key],size=num_samples)

        # Concatenate to the majority class
        X_resampled = np.concatenate((X_resampled,X[y == key],X[y == key][indx]), axis=0)
        y_resampled = np.concatenate((y_resampled, y[y == key], y[y == key][indx]))

    return X_resampled, y_resampled

