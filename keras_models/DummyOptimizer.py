from keras.optimizers import Optimizer
import sys


class DummyOptimizer(Optimizer):
    def get_updates(self, loss, params):
        print("get_updates was accidentally called")
        sys.exit(1)
