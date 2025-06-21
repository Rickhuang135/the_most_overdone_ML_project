import numpy as np

class Inference_results:
    def __init__(self, layer_values:list, activation_derivatives:list=None, cost_derivative=None):
        self.layer_values = layer_values
        if activation_derivatives is None:
            self.activation_derivatives = []
        else:
            self.activation_derivatives = activation_derivatives
        self.cost_derivative = cost_derivative
    def add(self, layer_value, activation_derivative):
        self.layer_values.append(layer_value)
        self.activation_derivatives.append(activation_derivative)
    def finalise(self, cost_derivative):
        self.cost_derivative = cost_derivative
    def __iter__(self):
        return Inference_results_iterator(self.activation_derivatives, self.layer_values)
    def latest(self):
        return self.layer_values[-1]
    
class Inference_results_iterator:
    def __init__(self, activation_derivatives, layer_values):
        self.activation_derivaties = activation_derivatives
        self.layer_values = layer_values
        self.current = len(layer_values)-3
    def __next__(self):
        if self.current >= 0:
            result = self.layer_values[self.current], self.activation_derivaties[self.current]
            self.current -= 1
            return result
        else:
            raise StopIteration()


class Network:
    def __init__(self, neurons: list[int], activation_functions: list, cost_function):
        self.layers = neurons
        self.activation_functions = activation_functions
        self.cost_function = cost_function
        self.parameters = [self.initialise_weights(x) for x in range(len(self.layers)-1)]
        self.og_parameters = [self.initialise_weights(x) for x in range(len(self.layers)-1)]
        self.biases = [np.zeros(x) for x in self.layers[1:]]

    def get_layer_shape(self, index:int):
        return (self.layers[index+1],self.layers[index])

    def initialise_weights(self, index: int):
        return np.random.randn(*self.get_layer_shape(index))*np.sqrt(2/self.layers[index])

    def inferance(self, data, label):
        results = Inference_results([data])
        for activation, parameters, bias in zip(self.activation_functions, self.parameters, self.biases):
            output, partial_derivatives = activation(parameters@results.latest()+bias)
            results.add(output, partial_derivatives)
        Cost, dC_dO = self.cost_function(results.latest(), label)
        results.finalise(dC_dO)
        return Cost, results
        

    def back(self, inference_results: Inference_results, step_size=0.01):
        gradients = []
        # dC_dlayers = [inference_results.cost_derivative]
        dC_dlayers = [inference_results.activation_derivatives[-1]@inference_results.cost_derivative]
        lastparameter_gradients = []
        for dC_dlastlayer in dC_dlayers[-1]:
            lastparameter_gradients.append(dC_dlastlayer*inference_results.layer_values[-2])
        gradients.append(np.array(lastparameter_gradients))

        for (layer_previous, activation_derivatives), parameters in zip (inference_results, self.parameters[::-1]):
            # dC_dlayers[-1] = parameters.T@dC_dlayers[-1]
            dC_dlayers.append((parameters@activation_derivatives).T@dC_dlayers[-1])
            temp = np.array([dC_dlayers[-1]])
            gradients.append(temp.T@np.array([layer_previous]))
        
        for index, gradient in enumerate(gradients[::-1]) :
            self.parameters[index] -= gradient * step_size

        for index, gradient in enumerate(dC_dlayers[::-1]):
            self.biases[index] -= gradient*step_size

    def compare_parameters(self):
        for index, (p1, p2) in enumerate(zip(self.parameters, self.og_parameters)):
            print(f"printing parameter set {index}")
            print(np.sort(p2-p1))

