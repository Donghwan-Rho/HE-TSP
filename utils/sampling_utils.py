import numpy as np

def f(x):
    return -2 * x**3 + 3 * x**2

def f1(x):
    return -1/2 * x**3 + 3/2 * x

g1_coeff1 = -1359 / (2 ** 10)
g1_coeff2 = 2126 / (2 ** 10)

def g1(x):
    result = g1_coeff1 * x**3 + g1_coeff2 * x
    return result

def psgn(x):
    return f1(f1(f1(g1(g1(x)))))

def heaviside(x):
    return (psgn(x)+1.0)*0.5
 
def amplifier(x):
    return g1(g1(g1(g1(g1(x)))))
 
def trigger(a,b):
    temp1 = 1-heaviside(a)
    temp2 = heaviside(b)
    return temp1*temp2

def trigger(a, b):
    a = np.float64(a)
    b = np.float64(b)
    temp1 = 1 - f(f(f(f(f(heaviside(a))))))
    temp2 = f(f(f(f(f(heaviside(b))))))
    return temp1 * temp2

def trigger_torch(a, b):
    temp1 = 1 - f(f(f(f(f(heaviside(a))))))
    temp2 = f(f(f(f(f(heaviside(b))))))
    return temp1 * temp2

def random_sampling(p, r):
    p = np.array(p, dtype=np.float64)
    r = np.float64(r)
    cumul_p = np.cumsum(p, dtype=np.float64)
    cumul_p_minus_r = cumul_p - r
    amplified = amplifier(amplifier(amplifier(cumul_p_minus_r)))
    prev_amplified = np.roll(amplified, 1)
    weight = trigger(prev_amplified, amplified)
    
    k = np.searchsorted(cumul_p, r)

    return weight, k

def post_processing(x):
    return -2 * x**3 + 3 * x**2
