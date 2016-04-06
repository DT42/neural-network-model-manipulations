import numpy as np
import theano
import theano.tensor as T
import time


def benchmark_matrix_dot_cpu_mode(d1, d2, d3, iteration):
    t0 = time.time()
    for i in xrange(iteration):
        m1 = np.random.rand(d1, d2)
        m2 = np.random.rand(d2, d3)
        m3 = np.dot(m1, m2)
    t1 = time.time()
    print('cpu time ({0},{1},{2}) {3} iterations: {4} secs'.format(
        d1, d2, d3, iteration, t1 - t0)
    )


def benchmark_matrix_dot_gpu_mode(d1, d2, d3, iteration):
    x = T.matrix('x')
    y = T.matrix('y')
    z = T.dot(x, y)
    f = theano.function([x, y], z, allow_input_downcast=True)

    t0 = time.time()
    for i in xrange(iteration):
        m1 = np.random.rand(d1, d2)
        m2 = np.random.rand(d2, d3)
        m3 = f(m1, m2)
    t1 = time.time()
    print('gpu time ({0},{1},{2}) {3} iterations: {4} secs'.format(
        d1, d2, d3, iteration, t1 - t0)
    )


# If the matrices are too small (400,20 x 20,1), GPU is slower than CPU
benchmark_matrix_dot_cpu_mode(400, 20, 1, 5000)
benchmark_matrix_dot_gpu_mode(400, 20, 1, 5000)

# If the matrices are large (5000,400 x 400,20), GPU is far fast than CPU
benchmark_matrix_dot_cpu_mode(5000, 400, 20, 100)
benchmark_matrix_dot_gpu_mode(5000, 400, 20, 100)
