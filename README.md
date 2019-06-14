# KerasMetaheuristicWrapper
Thin wrapper for keras to train models with various metaheuristic algorithms.

Uses two datasets as examples:
- iris (model has 67 trainable paramters)
- mnist (model has 13002 trainable paramters)

The example folders contain various script to train the model using SciPy, PySwarms and DEAP. The iris model can be trained this way reasonably fast. The mnist model proves to be too big to be solved by metaheuristics.
