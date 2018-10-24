import copy, numpy as np

# Compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# Convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

# Training dataset iteration
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary =  np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1
)
# Look-up table creation
for i in range(largest_number):
    int2binary[i] = binary[i]


# Input variables
alpha = 0.1 # Learning rate
input_dim = 2
hidden_dim = 16
output_dim = 1

# Initialize neural networks weights
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1 
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1 
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1 

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)


# Training logic
for j in range(10000):
    # Generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # True answer
    c_int = a_int + b_int
    c = int2binary[c_int]

    # Store for best guess (binary encoded)
    d = np.zeros_like(c)
    
    # Reset error messure to zero every time so that
    # the new bits can be seen how well they contribute to the prediction
    overallError = 0 

    layer_2_deltas = list() # Layer 2 derivatives
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim)) # Initialise the previous layer for the zeroth timestep


    # Moving along the positions in the binary encoding
    for position in range(binary_dim):
        # Generate input and output
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T # Will be either 1 or 0

        # Hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # Output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # Loss
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0]) # Sum of absolute errors to track propagation

        # Decode estimate so we can print it out; round the output to a binary value and store it in d
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # Store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    # Backpropagate from the last timestep
    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1] # Current hidden layer from the list
        prev_layer_1 = layer_1_values[-position - 2] # Previous hidden layer

        # Error at output layer
        layer_2_delta = layer_2_deltas[-position - 1]

        # Error at hidden layer
        # Current hidden layer error given the error at the hidden layer, from the future and at the current output layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)


        # Initialise weight update to try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    # Update weights using learning rate once backpropagation is done
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0


    # Print out progress
    if (j % 1000 == 0):
        print("Error: " + str(overallError))
        print("Pred: " + str(d))
        print("True: " + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("--------------------------------------------")
