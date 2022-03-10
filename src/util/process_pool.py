from multiprocessing import Pool
import multiprocessing


class ProcessPool:
    def __init__(self, n_processes=None):
        self.n_processes = n_processes if n_processes else multiprocessing.cpu_count()
 
    def run(self, callback_fn, params=[]):
        with Pool(processes=self.n_processes) as p:
            return p.starmap(callback_fn, params)