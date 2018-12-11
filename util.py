import sys
import time

from multiprocessing import Lock

iolock = Lock()

def annotated_timer(func_name):
    def timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            info = '='*80
            info += '\nFor function ##{}##, execute time is {:.4f} second\n'.format(func_name, end-start)
            info += '='*80+'\n'
            with iolock:
                sys.stdout.write(info)
            return result
        return wrapper
    return timer

def write_2out(format_str, *args):
    content = format_str.format(*args)
    with iolock:
        sys.stdout.write(content)

def write_2err(format_str, *args):
    content = format_str.format(*args)
    with iolock:
        sys.stderr.write(content)
