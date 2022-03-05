import pandas as pd

class RatingsMatrix:
    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def T(self):
        return RatingsMatrix(self.data.T)

    def row(self, row_id):
        return self.data.iloc[row_id-1, :].values.reshape(1, -1)

    def mean_row(self, row_id):
        return self.row(row_id).mean()
    
    def cell(self, row_id, col_id):
        return self.data.iloc[row_id-1, col_id-1]

    def row_deviation(self, row_id, col_id):
        return self.cell(row_id, col_id) - self.mean_row(row_id)
    
    @property
    def n_rows(self): return self.data.shape[0]
    
    @property
    def n_columns(self): return self.data.shape[1]

    def __repr__(self):
        display(self.data)
        return ""

    @property
    def cells(self):
        for user_idx in range(self.n_rows):
            user_id = user_idx + 1
            for item_idx in range(self.n_columns):
                item_id = item_idx + 1
                value = self.cell(user_id, item_id)
                yield (value, user_id, item_id)