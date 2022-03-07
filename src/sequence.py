class Sequence:
    def __init__(self, start=0):
        self.mapping = {}
        self.seq = start
    
    def apply(self, value):
        if value not in self.mapping:
            self.mapping[value] = self.seq
            self.seq += 1

        return self.mapping[value]
