__author__ = 'kilian'

import signal
import time
import resource
import random
import sys

class TimeoutError(Exception):
    pass

class MemoryoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        if self.seconds < sys.maxint:
            signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class memoryout:
    res = resource.RLIMIT_AS
    def __init__(self, kb=4096, error_message='Memoryout'):
        if kb != unlimited_memory:
            self.soft = kb * 1024
        else:
            self.soft = unlimited_memory
        self.error_message = error_message
    def handle_memoryout(self, signum, frame):
        raise MemoryoutError(self.error_message)
    def __enter__(self):
        # TODO: the signal handler does not seem to be effective,
        # TODO: instead a MemoryError is catched in run(..)
        if self.kb != unlimited_memory:
            signal.signal(signal.SIGSEGV, self.handle_memoryout)
            hard = resource.getrlimit(self.res)[1]
            print "Memory soft limit: ", self.soft
            resource.setrlimit(self.res, (self.soft, hard))
    def __exit__(self, type, value, traceback):
        if self.kb != unlimited_memory:
            hard = resource.getrlimit(self.res)[1]
            resource.setrlimit(self.res, (hard, hard))
            print "Memory soft limit R", hard

unlimited_memory = resource.RLIM_INFINITY

def run(f, time_out, mb, *args):
    try:
        with timeout(seconds=time_out):
            if mb != unlimited_memory:
                kb = kb=1024 * mb
            else:
                kb = unlimited_memory
            with memoryout(kb=kb):
                return f(*args)
    except (TimeoutError, MemoryoutError, MemoryError) as e:
        if isinstance(e, MemoryError):
            hard = resource.getrlimit(resource.RLIMIT_AS)[1]
            resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
            print "Memory soft limit R", hard
            return MemoryoutError('Memoryout')
        return e

def f(x):
    time.sleep(x)
    i = 0
    content = []
    for i in range(2000000):
        content.append(random.random())
    return len(content)

if __name__ == '__main__':
    list = [3, 1.0, 1.9]
    list2 = map(lambda x: run(f, x, 2), list)
    for x in list2:
        print x
    print "Usage:", resource.getrusage(resource.RUSAGE_SELF)
    print "Limit AS:", resource.getrlimit(resource.RLIMIT_AS)
