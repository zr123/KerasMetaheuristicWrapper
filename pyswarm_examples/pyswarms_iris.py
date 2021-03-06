import timeit

import numpy as np
import pyswarms as ps
from keras_models import iris
from wrapper.Wrapper import Wrapper

# Load model and datasaet
model = iris.get_model()
(x_train, y_train) = iris.get_data()

# Initialize swarm
wrapper = Wrapper(model, x=x_train, y=y_train)

# Set hyperparameters c1: cognitive component , c2: social component, w: inertia
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# set bounds
max_bound = np.ones(wrapper.get_dimensions())
min_bound = - max_bound
bounds = (min_bound, max_bound)

# Create PySwarm optimizer
optimizer = ps.single.GlobalBestPSO(
    n_particles=100,
    dimensions=wrapper.get_dimensions(),
    options=options,
    bounds=bounds)

start = timeit.default_timer()

# Perform optimization
cost, pos = optimizer.optimize(wrapper.get_all_fitnesses, iters=100)

loss, acc = wrapper.evaluate(pos, x_train, y_train)
print("Total loss: ", loss, "total accuracy: ", acc)
print('Runtime: ', timeit.default_timer() - start, "seconds ")


exit()
