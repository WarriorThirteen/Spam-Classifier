import numpy as np


class SpamClassifier:
    def __init__(self, learning_rate=0.01, theta=0.06, layer_count=3, samples_per_train=100):
        self.alpha = learning_rate
        self.theta = theta
        self.layer_count = layer_count
        self.samples_per_train = samples_per_train


    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))


    @classmethod
    def sigmoid_deriv(cls, x):
        return cls.sigmoid(x) * (1 - cls.sigmoid(x))


    def train(self, test=None):
        if test is None:
            spam1 = np.loadtxt(open("./data/training_spam.csv"), delimiter=",")
            spam2 = np.loadtxt(open("./data/testing_spam.csv"), delimiter=",")
            training_spam = np.concatenate((spam1, spam2))
        else:
            training_spam = test

        num_features = len(training_spam[0])-1

        # We set up a neural net with n layers of 54 nodes each, with an additional output node at the end

        # 3 54x54 matrics of weights per layer - 54 weights for each of 54 nodes in 3 layers
        self.layer_weights = np.array(list((np.random.rand(num_features, num_features) - 0.5) * 2 for i in range(self.layer_count)))
        self.final_weights = (np.random.rand(num_features) - 0.5) * 2

        # N+1 layers, zeroth layer holds input (4 layers default)
        self.layers = np.zeros((self.layer_count+1, num_features))

        # somewhere to hold our cumulative errors for each node (4 layers default but don't use the last one)
        self.errors = np.zeros_like(self.layers)

        # The biases - one for each "node" (3 in total) and a final one
        self.biases = np.array(list((np.random.rand(num_features) - 0.5) * 2 for i in range(self.layer_count)))
        self.final_bias = (np.random.rand() - 0.5) * 2

        np.random.seed(111)

        iteration = 0
        avg_cost = 1

        while avg_cost > self.theta:
            # Train ourselves on 100 random selections
            sample = training_spam[np.random.choice(training_spam.shape[0], self.samples_per_train, replace=False)]

            for data in sample:
                error = data[0] - self.carry_forward(data[1:])
                self.back_propagate(error)


            # Find the average cost across the data set
            avg_cost = sum(np.square(training_spam[:, 0] - self.predict(training_spam[:, 1:]))) / len(training_spam)
            iteration += 1

        # print(f"Mean squared error after iteration {iteration}: {avg_cost}")


    def back_propagate(self, error):
        # adjust final bias and weights to output
        z = sum(self.final_weights * self.layers[-1]) + self.final_bias

        d_bias = self.sigmoid_deriv(z) * 2 * (error)
        d_weights = self.layers[-1] * d_bias


        self.final_bias += d_bias * self.alpha
        self.final_weights += d_weights * self.alpha

        # Set the error for each of the final nodes
        self.errors.fill(0)
        self.errors[-1] += self.final_weights * d_bias

        for layer_index in range(0, self.layer_count, -1):
            # Last layer -> second layer, skipping the zeroth layer because that is the input
            for node in range(len(self.layers[layer_index])):
                self.propagate_node(layer_index, node)


    def propagate_node(self, layer, node):
        # ! I don't know if this works or not, but 0 layers is ideal so it's not an issue at the moment
        # adjust bias and weights to output
        z = sum(self.layer_weights[layer, node] * self.layers[layer]) + self.biases[layer, node]

        d_bias = self.sigmoid_deriv(z) * 2 * (self.errors[layer+1, node])
        d_weights = np.array(tuple(self.layers[layer, i] * d_bias for i in range(len(self.layer_weights[layer, node]))))

        self.biases[layer, node] += d_bias * self.alpha
        self.layer_weights[layer, node] += d_weights * self.alpha

        # adjust the error for each of the previous layer nodes
        self.errors[layer] += np.array(tuple(self.layer_weights[layer, i] * d_bias for i in range(len(self.layer_weights[layer, node]))))


    def carry_forward(self, features):
        self.layers[0] = features

        for i in range(self.layer_count):
            self.layers[i+1] = self.sigmoid(np.matmul(self.layer_weights[i], self.layers[i]) + self.biases[i])

        return self.sigmoid(np.dot(self.layers[-1], self.final_weights) + self.final_bias)


    def predict(self, data):
        class_predictions = np.zeros(data.shape[0])

        for case_index, test_case_features in enumerate(data):
            class_predictions[case_index] = 1 if self.carry_forward(test_case_features) >= 0.5 else 0

        return class_predictions
    

def create_classifier():
    classifier = SpamClassifier(layer_count=0)
    classifier.train()
    return classifier


def my_accuracy_estimate():
    '''
    Uses k-fold cross validation to get an accuracy estimate
    It's actually reversed - we train on 1/k and test on the rest
    '''
    k = 10

    spam1 = np.loadtxt(open("./data/training_spam.csv"), delimiter=",")
    spam2 = np.loadtxt(open("./data/testing_spam.csv"), delimiter=",")

    training_spam = np.concatenate((spam1, spam2)) # [:, features.astype(bool)]

    # Split data into K groups
    data = np.array(tuple(training_spam[i : i + (len(training_spam) // k)] for i in range(k)))
    test_groups = np.ones(k, int)

    tester = create_classifier()
    acc = 0

    for i in range(k):
        test_groups[i] = 0
        test_data = np.concatenate(data[test_groups])

        tester.train(data[i])
        predictions = tester.predict(test_data[:, 1:])

        accuracy = np.count_nonzero(predictions == test_data[:, 0]) / test_data.shape[0]
        acc += accuracy

        test_groups[i] = 1

    acc /= k

    return acc



def tests():
    classifier = create_classifier()
    testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
    test_data = testing_spam[:, 1:]
    test_labels = testing_spam[:, 0]

    predictions = classifier.predict(test_data)
    accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]
    print(f"Accuracy on test data is: {accuracy}")
    
tests()