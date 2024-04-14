import random
import string
import os
import math


class Perceptron:
    def __init__(self, language, num_inputs=26, learning_rate=0.01):
        self.weights = {letter: random.uniform(-1, 1) for letter in string.ascii_lowercase}
        self.bias = 0
        self.learning_rate = learning_rate
        self.language = language
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    def predict(self, inputs: dict):
        weighted_sum = self.bias

        for key in inputs:
            if key != 'desired_output':
                # print("Inputs[key] " + str(inputs[key]) + " self.weights[key] " + str(self.weights[key]))
                weighted_sum += inputs[key] * self.weights[key]
                # print(weighted_sum)

        prediction = 1 if weighted_sum > 0 else 0
        return prediction

    def train(self, vectors):
        for _ in range(1000):
            for vector in vectors:
                prediction = self.predict(vector)
                vector_language = vector['desired_output']
                desired_output = 0
                if vector_language == self.language:
                    desired_output = 1

                for key in vector:
                    if key != 'desired_output':
                        self.weights[key] += self.learning_rate * (desired_output - prediction) * vector[key]

                self.bias += self.learning_rate * (desired_output - prediction)


class Neural_network:
    def __init__(self):
        self.german_perceptron = Perceptron(language='German')
        self.english_perceptron = Perceptron(language='English')
        self.polish_perceptron = Perceptron(language='Polish')

    def train(self):
        self.german_perceptron.train(self.training_data)
        self.english_perceptron.train(self.training_data)
        self.polish_perceptron.train(self.training_data)

    def test(self):
        correctly_predicted = 0

        for i in os.listdir("testing_data"):
            folder_path = os.path.join("testing_data", i)
            for j in os.listdir(folder_path):
                file_path = os.path.join(folder_path, j)
                letters_count = self.parse_file_to_dict(file_path)
                results = {
                    'German': self.german_perceptron.predict(letters_count),
                    'English': self.english_perceptron.predict(letters_count),
                    'Polish': self.polish_perceptron.predict(letters_count)
                }
                print(results)
                path = file_path.split("\\")
                language = path[-2]
                for key, value in results.items():
                    if key == language and value == 1:
                        correctly_predicted += 1
                        break

        print("Accuracy " + str((correctly_predicted * 100) / 9))



    def load_training_data(self):
        self.training_data = []

        for i in os.listdir("training_data"):
            folder_path = os.path.join("training_data", i)
            for j in os.listdir(folder_path):
                file_path = os.path.join(folder_path, j)
                lettes_count = self.parse_file_to_dict(file_path)
                self.training_data.append(lettes_count)
                print(lettes_count)

    def parse_file_to_dict(self, file_path):
        letters_count = {letter: 0 for letter in string.ascii_lowercase}
        counter = 0

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                for letter in line:
                    lowercase_letter = letter.lower()
                    if lowercase_letter in letters_count:
                        counter += 1
                        letters_count[lowercase_letter] += 1


        for key, value in letters_count.items():
            letters_count[key] = value / counter

        path = file_path.split('\\')
        language = path[-2]
        letters_count['desired_output'] = language

        return letters_count


if __name__ == '__main__':
    neural_net = Neural_network()
    neural_net.load_training_data()
    neural_net.test()
