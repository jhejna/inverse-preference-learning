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

    def close(self):
        # By default do nothing
        return

class TensorBoardWriter(Writer):

    def __init__(self, path):
        super().__init__(path)
        self.writer = SummaryWriter(self.path)

    def dump(self, step):
        for k in self.values.keys():
            self.writer.add_scalar(k, self.values[k], step)
        self.writer.flush()
        self.values.clear()

    def close(self):
        self.writer.close()

class CSVWriter(Writer):

    def __init__(self, path):
        super().__init__(path)
        self._csv_file_handler = None
        self.csv_logger = None
        self.num_keys = 0
        
    def _reset_csv_handler(self):
        if self._csv_file_handler is not None:
            self._csv_file_handler.close() # Close our fds
        self.csv_file_handler = open(os.path.join(self.path, "log.csv"), "w")
        self.csv_logger = csv.DictWriter(self.csv_file_handler, fieldnames=list(self.values.keys()))
        self.csv_logger.writeheader()

    def record(self, key, value):
        super().record(key, value)

    def dump(self, step):
        # Record the step
        self.values["step"] = step
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

    def close(self):
        self.csv_file_handler.close()

class WandBWriter(Writer):

    def __init__(self, path):
        super().__init__(path)
        self.num_keys = 0
        
    def dump(self, step):
        # Only log if we have all of the values -- this makes it more similar to the csv
        # This is done to prevent syncing to WandB constantly.
        if len(self.values) < self.num_keys:
            return
        elif len(self.values) > self.num_keys:
            self.num_keys = len(self.values)
        wandb.log(self.values, step=step)
        self.values = {} # reset the values

    def close(self):
        wandb.finish()

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

    def close(self):
        for writer in self.writers:
            writer.close()
