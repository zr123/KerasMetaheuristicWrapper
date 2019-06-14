import timeit
from keras_models import mnist


# Load model and datasaet
model = mnist.get_model()
(x_train, y_train, x_test, y_test) = mnist.get_data()

# Recompile model without dummy optimzer
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

start = timeit.default_timer()

# Train model
model.fit(x_train, y_train, batch_size=32, epochs=10)

# Evaluate model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Total loss: ", loss, "total accuracy: ", acc)
stop = timeit.default_timer()
print('Runtime: ', stop - start, "seconds ")

exit()
