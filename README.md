
# Profile

CPU mode

    ...

Most time will be spent in numpy.core.multiarray.dot

GPU mode

    $ python -m cProfile -o gpu.log <>
    []: import pstats
    []: p = pstats.Stats('gpu.log')
    []: p.sort_stats('tottime').print_stats(10)

Most time will be spent in Theano's compiled function.


The matrix dot will be called around 1.2M times.
