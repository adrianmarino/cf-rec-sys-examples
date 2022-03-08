from multiprocessing import Pool


class ProcessPool:
    def __init__(self, n_processes): self.n_processes = n_processes
 
    def run(self, callback_fn, params=[]):
        with Pool(processes=self.n_processes) as p:
            return p.starmap(callback_fn, params)