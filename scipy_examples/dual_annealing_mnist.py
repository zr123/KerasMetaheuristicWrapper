import timeit

from scipy import optimize
from keras_models import mnist
from wrapper.Wrapper import Wrapper

# Load model and datasaet
model = mnist.get_model()
(x_train, y_train, x_test, y_test) = mnist.get_data()

wrapper = Wrapper(model, x_train, y_train)

bounds = [(-1.0, 1.0)] * wrapper.get_dimensions()

start = timeit.default_timer()

pos = optimize.dual_annealing(
    wrapper.get_fitness,
    bounds=bounds,
    maxiter=1)

loss, acc = wrapper.evaluate(pos.x, x_test, y_test)
print("Total loss: ", loss, "total accuracy: ", acc)
print('Runtime: ', timeit.default_timer() - start, "seconds ")

exit()
