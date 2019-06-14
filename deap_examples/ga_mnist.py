import random
import timeit
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from keras_models import mnist
from wrapper.Wrapper import Wrapper

# Load model and datasaet
model = mnist.get_model()
(x_train, y_train, x_test, y_test) = mnist.get_data()

# Initialize swarm
wrapper = Wrapper(model, x=x_train, y=y_train)

# register a minimization-problem with a single fitness-value and use a numpy-array as individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMin)

# register population initializer
toolbox = base.Toolbox()
toolbox.register("random_float", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.random_float, n=wrapper.get_dimensions())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# register required functions of eaSimple
toolbox.register("evaluate", wrapper.my_fitenss_as_tuple)
toolbox.register("mate", wrapper.crossover)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

start = timeit.default_timer()
# perform GA
hof = tools.HallOfFame(1, similar=numpy.array_equal)
algorithms.eaSimple(
    population=toolbox.population(n=100),   # set population
    toolbox=toolbox,                        # hand over toolbox
    cxpb=0.5,                               # crossover probability
    mutpb=0.2,                              # mutation probability
    ngen=100,                               # number of generations (iterations)
    halloffame=hof,                         # remember the single best individual
    verbose=True                            # display progress
)

loss, acc = wrapper.evaluate(hof.items[0], x_test, y_test)
print("Total loss: ", loss, "total accuracy: ", acc)
print('Runtime: ', timeit.default_timer() - start, "seconds ")

exit()