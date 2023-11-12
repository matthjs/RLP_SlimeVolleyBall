"""
online algorithms for keeping track of sample means and sample variances
"""
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def updateAggregate(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalizeAggregate(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        # float("nan")
        return (mean, 0, 0)
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)