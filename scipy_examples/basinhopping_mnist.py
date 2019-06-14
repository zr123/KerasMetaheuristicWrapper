import timeit

from scipy import optimize
from keras_models import mnist
from wrapper.Wrapper import Wrapper

# Load model and datasaet
model = mnist.get_model()
(x_train, y_train, x_test, y_test) = mnist.get_data()

# Initialize swarm
wrapper = Wrapper(model, x=x_train, y=y_train)

start = timeit.default_timer()

pos = optimize.basinhopping(
    wrapper.get_fitness,
    wrapper.get_chromosome(),
    niter=1)

loss, acc = wrapper.evaluate(pos.x, x_test, y_test)
print("Total loss: ", loss, "total accuracy: ", acc)
print('Runtime: ', timeit.default_timer() - start, "seconds ")

exit()
