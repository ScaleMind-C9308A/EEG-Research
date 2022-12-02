import scipy.stats

def binomial_cmf(k, n, p):
    c = 0
    for k1 in range(n+1):
        if k1>=k:
            c += scipy.stats.binom.pmf(k1, n, p)
    return c

def p_value(accuracy, samples, n_classes):
    return binomial_cmf(round(accuracy*samples), samples, 1.0/n_classes)

def significance(p_value, samples, n_classes):
    for k in range(samples/n_classes-1, samples+1):
        if binomial_cmf(k, samples, 1.0/n_classes)<p_value:
            return float(k)/samples
