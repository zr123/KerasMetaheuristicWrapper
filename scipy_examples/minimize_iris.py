import timeit

from scipy import optimize
from keras_models import iris
from wrapper.Wrapper import Wrapper

# Load model and datasaet
model = iris.get_model()
(x_train, y_train) = iris.get_data()

wrapper = Wrapper(model, x_train, y_train)

start = timeit.default_timer()

pos = optimize.minimize(
    wrapper.get_fitness,
    wrapper.get_chromosome(),
    method='COBYLA')

loss, acc = wrapper.evaluate(pos.x, x_train, y_train)
print("Total loss: ", loss, "total accuracy: ", acc)
print('Runtime: ', timeit.default_timer() - start, "seconds ")

exit()
