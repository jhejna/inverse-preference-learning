import os
import csv
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except:
    pass

class Writer(ABC):

    def __init__(self, path):
        self.path = path
        self.values = {}

    def record(self, key, value):
        self.values[key] = value
    
    @abstractmethod
    def dump(self, step):
        return NotImplementedError

class TensorBoardWriter(Writer):

    def __init__(self, path):
        super().__init__(path)
        self.writer = SummaryWriter(self.path)

    def dump(self, step):
        for k in self.values.keys():
            self.writer.add_scalar(k, self.values[k], step)
        self.writer.flush()
        self.values.clear()

class CSVWriter(Writer):

    def __init__(self, path):
        super().__init__(path)
        self.updated = {}
        self._csv_file_handler = None
        self.csv_logger = None
        self.num_keys = 0
        
    def _reset_csv_handler(self):
        if self._csv_file_handler is not None:
            self._csv_file_handler.close() # Close our fds
        self.csv_file_handler = open(os.path.join(self.path, "log.csv"), "wt")
        self.csv_logger = csv.DictWriter(self.csv_file_handler, fieldnames=list(self.values.keys()))
        self.csv_logger.writeheader()

    def record(self, key, value):
        super().record(key, value)
        self.updated[key] = True

    def dump(self, step):
        # Record the step
        self.values["step"] = step
        self.updated["step"] = True
        # Check if we need to re-create the csv writer
        if len(self.values) < self.num_keys:
            return # Do nothing, we don't have all the keys yet.
        elif len(self.values) > self.num_keys:
            # Get got a new key, so re-create the writer
            self.num_keys = len(self.values)
            self._reset_csv_handler()
        # We should now have all the keys
        self.csv_logger.writerow(self.values)
        self.csv_file_handler.flush()
        self.values = {} # Reset the values.

class WandBWriter(Writer):

    def __init__(self, path):
        super().__init__(path)
        
    def dump(self, step):
        wandb.log(self.values, step=step)
        self.values = {} # reset the values

class Logger(object):

    def __init__(self, path, writers=['tb', 'csv']):
        self.writers = []
        for writer in writers:
            self.writers.append(
                {
                    'tb': TensorBoardWriter,
                    'csv': CSVWriter,
                    'wandb': WandBWriter
                }[writer](path)
            )

    def record(self, key, value):
        for writer in self.writers:
            writer.record(key, value)

    def dump(self, step):
        for writer in self.writers:
            writer.dump(step)
