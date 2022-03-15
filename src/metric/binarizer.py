class identity:
    def __init__(self, desc=''): self._desc = desc
    @property
    def desc(self): return self._desc
    def closure(self): return lambda it: it

class gte:
    def __init__(self, value): self._value = value
    @property
    def desc(self): return f' (R>={self._value})'
    def closure(self): return lambda it: 1 if it >= self._value else 0


class between:
    def __init__(self, begin, end): 
        self.begin = begin
        self.end = end

    @property
    def desc(self): return f'({self.begin},{self.end})'
    def closure(self): return lambda it: self.begin <= it <= self.end 