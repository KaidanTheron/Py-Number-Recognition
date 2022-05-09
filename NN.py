"""
Feel free to use this project as a learning resource for yourself. Unfortunately I can't promise that my train of thought
will be completely understandable when looking at the code, but I will try to comment and explain my code as much as
possible.

If you are going to use this as a learning resource, I highly advise that you don't make it your main learning resource. 
This project will help most if you are using it in conjunction with other resources such as the youtube series on machine
learning by 3Blue1Brown.

First, I will make and use a vanilla ml model to recognize the digits, then I will compare the accuracy of the vanilla 
model to the accuracy of the CNN I will be making and using to recognize the digits. The CNN should yield more 
accurate results compared to the vanilla NN.

So, what do we need for the NN?
We need layers, nodes within those layers and connections between the nodes in those layers. 
I will be using lists to store the values of the weights of the connections, also to store all of the neurons which
I will be representing by using objects that store the neuron's activation and its unique bias.

Each layer is comprised of nodes, or neurons if you will. Each node is connected to each node in the previous layer.
A node has a bias, a weighed input from each previous node, and a activation function. I will be using sigmoid
activation function. For a given node, if it isn't an input node, its activation will be a sum of the activations of each
node in the previous layer, times the weight of their connections.

A node is either lit up, e.g 1.00 or off, e.g. 0.00. It can also be any 2 decimal, real value in between, e.g. 0.54.

I will also be using pygame as my library of choice to provide a visual interface. This visualisation takes place
in the main.py file.

This project is heavily inspired by the content that can be found in 3Blue1Brown's youtube series about machine learning
and neural networks.

Last note: mind you, I am making this completely from scratch, and I haven't looked at the code of any machine learning
library out there. This means that my code is based solely on my understanding of the concepts and mathematics that machine
learning is based on. The goal of this project isn't to make the best, most efficient machine learning library out there,
the goal is to write everything from scratch as to cement the fundamentals in my mind and know what is going on inside
the pre-written libraries for machine learning that already exist, because it is likely I will be using them some day.
In a perfect world I would always be writing everything I use from scratch, that's just the way I like it (its also the
best way to learn), but this simply isn't time efficient. Naturally, I would really like to encourage you to try this
yourself. When you finish it and you take a step back and look at what you have created you will feel very satisfied with
how much you have learned in the process.

-16 April 2021-
"""
import random
import math
from keras.datasets import mnist # importing the handwritten set of digits from mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data() # load the data from the mnist database to train and test on

class NN(object):

    # Define a general template for each neuron in the network
    class node(object):
        def __init__(self, bias_lower, bias_upper, is_random):
            if is_random:
                self.activation = 0.0 # this is the value the node outputs when it receives an input
                self.bias = random.randint(bias_lower, bias_upper)
            else:
                self.activation = 0.0
                self.bias = 0
        
        # Activation function: (squishes all real numbers into the interval 0 and 1)
        def sigmoid(self, x):
            return round(1/(1+math.e**(-x)), 2)

        # Calculate and set this neurons activation
        def activate(self, weighted_sum): # input to a node/neuron must be weighted sum of previous nodes
            self.activation = self.sigmoid(weighted_sum - self.bias)

    # user must be able to specify the number of layers, neurons in each of those nodes, and the range
    # of the unique random value each weight and bias will start as
    def __init__(self, num_layers=4, num_nodes=[28*28, 16, 16, 10], rand_bias=[-10, 10], rand_weight=[-10, 10]):
        self.layers = []
        self.connections = []
        self.targets = [] # stores the values of what each output neuron should be for given image - used during back prop
        
        # Create number of layers that user specifies
        for i in range(num_layers):
            # User inputs a number of nodes for each layer
            # Create that number of nodes, each with a random bias in a range that the user specified
            self.layers.append([self.node(rand_bias[0], rand_bias[1], True) for _ in range(num_nodes[i])])
        
        # Now we create the connections (weights) that connect each node to the next
        # Firstly, there will be a list for each group of connections between two layers
        # Within these groups there will be a list of connections (weights) for each node that corresponds to its
        # connections to every node in the previous layer.
        for i in range(1, num_layers):
            self.connections.append([])
            for j in range(num_nodes[i]):
                self.connections[i-1].append([])
                for _ in range(num_nodes[i-1]):
                    self.connections[i-1][j].append(random.uniform(rand_weight[0], rand_weight[1]))
        
    # This function receives an input of pixels and makes each the activation of a node in the input (first) layer of 
    # the NN
    def initialize_inputs(self, input_list):
        for i in range(len(input_list)):
            self.layers[0][i].activation = round(1/(1+math.e**(-input_list[i])), 2)
    
    # This function handles giving the inputs and then carrying the corresponding output from each layer of nodes
    # to the next.
    def feed_forward(self): # inputs is a list of inputs with the same amount of elements as there are nodes
                            # in the first layer of the network, this is because the first layer of nodes
                            # are, of course, the input nodes
        for layer_index in range(1, len(self.layers)): # exclude the first layer because it is the input layer and is already accounted for
            for node_index in range(len(self.layers[layer_index])):
                weighted_sum = 0
                for connection_index in range(len(self.connections[layer_index-1][node_index])):
                    weighted_sum += self.connections[layer_index-1][node_index][connection_index]*self.layers[layer_index-1][connection_index].activation
                self.layers[layer_index][node_index].activate(weighted_sum)

    # Here we give the NN a random image and see how well it performs    
    def test(self):
        # THe code below picks a random number and puts each pixel value of that number from the MNIST digits, in a list to
        # be fed to the NN
        random_number_index = random.randint(0, 10000)
        random_number = test_y[random_number_index]
        pixels = []
        for i in range(len(test_X[random_number_index])):
            for j in range(len(test_X[random_number_index][i])):
                pixels.append(test_X[random_number_index][i][j])
        self.initialize_inputs(pixels)
        self.feed_forward()

        # Display the NN's guess for an image
        print(f'The correct number is %d, and the NN guessed the following for each number:'%(random_number))
        for i in range(len(self.layers[3])):
            print(f'The NN is %f sure it is %d'%(self.layers[3][i].activation, i))

    # This function returns the gradient vector of the cost function so we can take a step for each bias and weight that
    # will decrease the value of the cost function     
    # 
    # Back propagation is possibly the hardest part of machine learning to understand in the start, I would recommend
    # having a firm understanding of it before looking at this code  
    def back_prop(self, image_cost):
        weight_gradient_adjustments = [[[0 for _ in range(len(self.connections[i][j]))] for j in range(len(self.connections[i]))] for i in range(len(self.connections))]
        bias_gradient_adjustments = [[0 for _ in range(len(self.layers[i+1]))] for i in range(len(self.layers)-1)]
        # this matrix is essential to the back propagation, we will take the error from the output
        # || neurons and then propagate it backwards until we have an error calculation for each neuron
        # || except for the input neurons
        # \/
        neuron_errors = [[0 for _ in range(len(self.layers[i+1]))] for i in range(len(self.layers)-1)]
        for i in range(len(neuron_errors[len(neuron_errors)-1])):
            # this next line looks crazy, and
            neuron_errors[len(neuron_errors)-1][i] = (self.layers[len(self.layers)-1][i].activation-self.targets[i])*((self.layers[len(self.layers)-1][i].activation)*(1-self.layers[len(self.layers)-1][i].activation))

        # because of the way my weight matrix is structured, I will be doing some funny operations below
        # excuse the lack of sensibility/readability in advance
        for i in range(len(neuron_errors)-2, -1, -1):
            for j in range(len(neuron_errors[i])):
                neuron_error = 0
                for k in range(len(neuron_errors[i+1])):
                    neuron_error += (self.connections[i+1][k][j])*(neuron_errors[i+1][k])
                neuron_errors[i][j] = neuron_error*((self.layers[i+1][j].activation)*(1-self.layers[i+1][j].activation))

        # here we find the gradient of the cost function relative to each weight and then add that to our list called
        # weight_gradient_adjustments
        # the gradient of the cost function relative to each weight is the product of the activation of the neuron it is
        # coming from, and the error of the neuron it is connecting to
        for i in range(len(weight_gradient_adjustments)):
            for j in range(len(weight_gradient_adjustments[i])):
                for k in range(len(weight_gradient_adjustments[i][j])):
                    weight_gradient_adjustments[i][j][k] = self.layers[i][k].activation*neuron_errors[i][j]

        # here we find the gradient of the cost function relative to each bias and then add that to our list called
        # bias_gradient_adjustments
        # lucky for us - well, actually me - the derivative of the cost function in relation to a given bias is just
        # the error of the neuron
        for i in range(len(neuron_errors)):
            for j in range(len(neuron_errors[i])):
                bias_gradient_adjustments[i][j] = neuron_errors[i][j]
                    
        return weight_gradient_adjustments, bias_gradient_adjustments

    # This function takes in the gradient vectors calculated by back prop, then multiplies each element by the step size
    # and steps each corresponding weight or bias in the direction that decreases the cost function
    # in other words we subtract the proportional step from the existing 
    def gradient_descent(self, step_size, weight_gradient, bias_gradient):
        for i in range(len(self.connections)):
            for j in range(len(self.connections[i])):
                for k in range(len(self.connections[i][j])):
                    self.connections[i][j][k] -= weight_gradient[i][j][k]*step_size

        for i in range(len(self.layers)-1):
            for j in range(len(self.layers[i+1])):
                self.layers[i+1][j].bias -= bias_gradient[i][j]*step_size

    #This function will return how wrong each output neuron is for a given image
    def find_image_cost(self, image_label):
        image_cost = 0
        for node_index in range(len(self.layers[len(self.layers)-1])):
            if node_index == image_label:
                # if this is the neuron that is correct, we want it to be lit up all the way i.e, having an activation of
                # 1, so we find the cost for this neuron by squaring the difference between it's activation and 1
                self.targets.append(1)
                image_cost += (1-self.layers[len(self.layers)-1][node_index].activation)**2
            else:
                # if this neuron isn't the right answer we want it to be off, i.e. have an activation of 0
                # so we find the cost for this neuron by squaring the difference between it's activation and 0
                self.targets.append(0)
                image_cost += (0-self.layers[len(self.layers)-1][node_index].activation)**2
        return 0.5*(image_cost)

    #This function will add the elements of the weight_gradient and the weight_gradient_adjustments
    def add_weight_gradient_matrices(self, matrix1, matrix2):
        return [[[(matrix1[i][j][k]+matrix2[i][j][k]) for k in range(len(matrix1[i][j]))] for j in range(len(matrix1[i]))] for i in range(len(matrix1))]

    #This function will add the elements of the bias_gradient and the bias_gradient_adjustments
    def add_bias_gradient_matrices(self, matrix1, matrix2):
        return [[(matrix1[i][j]+matrix2[i][j]) for j in range(len(self.layers[i+1]))] for i in range(len(self.layers)-1)]

    # This is the function where the learning happens
    # We give the NN a set of training data, see what is guesses, find how wrong it is, and then adjust the weights and
    # biases accordingly
    # in other words finds cost to use back propogation to find gradient, then uses gradient descent on that gradient
    def train(self, train_data, mini_batch_size=100, step_size=100): # why a step size? it helps avoid overshooting
        for batch_index in range(len(train_data) // mini_batch_size):
            # create zero filled matrices for the gradient vector of the biases and the gradient vector of the weights
            weight_gradient = [[[0 for _ in range(len(self.connections[i][j]))] for j in range(len(self.connections[i]))] for i in range(len(self.connections))]
            bias_gradient = [[0 for _ in range(len(self.layers[i+1]))] for i in range(len(self.layers)-1)]
            average_cost = 0

            for number_image_index in range(batch_index*mini_batch_size, batch_index*mini_batch_size+mini_batch_size):
                image_label = train_y[number_image_index]
                pixels = []
                for i in range(len(train_X[number_image_index])):
                    for j in range(len(train_X[number_image_index][i])):
                        pixels.append(train_X[number_image_index][i][j])

                # give the NN the pixels and see what it thinks
                network.initialize_inputs(pixels)
                network.feed_forward()

                # in this step we find how wrong the NN is and then calculate the gradient of the cost function for
                # this image
                image_cost = self.find_image_cost(image_label)
                average_cost += image_cost
                weight_gradient_adjustment, bias_gradient_adjustment = self.back_prop(image_cost)
                self.add_weight_gradient_matrices(weight_gradient, weight_gradient_adjustment)
                self.add_bias_gradient_matrices(bias_gradient, bias_gradient_adjustment)

            average_cost /= mini_batch_size
            print(average_cost)

            # Before applying gradient descent we need to average all the elements of the weight_gradient and 
            # bias_gradient vectors since we added the gradients from each image in this mini batch
            # weight_gradient = [[[(weight_gradient[i][j][k]/mini_batch_size) for k in range(len(weight_gradient[i][j]))] for j in range(len(weight_gradient[i]))] for i in range(len(weight_gradient))]
            # bias_gradient = [[(bias_gradient[i][j]/mini_batch_size) for j in range(len(bias_gradient[i]))] for i in range(len(bias_gradient))]
 
            # apply gradient descent for this batch
            self.gradient_descent(step_size, weight_gradient, bias_gradient) 
        network.test()

network = NN()
network.train(train_X)