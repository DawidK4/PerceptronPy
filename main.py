import random
import string
import os
import numpy as np

class Perceptron:
    def __init__(self, num_inputs=26, learning_rate=0.01):
        self.weights = {letter: random.uniform(-1, 1) for letter in string.ascii_lowercase}
        self.bias = 0
        self.learning_rate = learning_rate

    def activation_fun(self, x):
        return 1.0 / (1.0 + np.exp(-x)) # why sometimes returned value is grater that 0???

    def predict(self, inputs: dict, train: bool):
        weighted_sum = self.bias

        for key in inputs:
            if key != 'desired_output':
                weighted_sum += inputs[key] * self.weights[key]

        if train:
            prediction = 1 if weighted_sum > 0 else 0
            return prediction

        prediction = self.activation_fun(weighted_sum)
        return prediction



    def train(self, vectors):
        for _ in range(1000):
            for vector in vectors:
                prediction = self.predict(vector, True)
                vector_language = vector['desired_output']
                desired_output = 0
                if vector_language == self.language:
                    desired_output = 1

                for key in vector:
                    if key != 'desired_output':
                        self.weights[key] += self.learning_rate * (desired_output - prediction) * vector[key]

                self.bias += self.learning_rate * (desired_output - prediction)


class NeuralNetwork:
    def __init__(self):
        self.perceptrons = []

    def predict(self):
        text = str(input('Enter text: '))
        letters_count = {letter: 0 for letter in string.ascii_lowercase}
        text_length = 0
        for letter in text:
            lowercase_letter = letter.lower()
            if lowercase_letter in letters_count:
                letters_count[lowercase_letter] += 1
                text_length += 1

        for key, value in letters_count.items():
            letters_count[key] = (letters_count[key] * 100) / text_length

        results = {perceptron.language: perceptron.predict(letters_count, False) for perceptron in self.perceptrons}
        print(results)
        print("Prediction: " + str(max(results, key=results.get)))

    def train(self, training_data):
        languages = set()
        for data in training_data:
            languages.add(data['desired_output'])

        for language in languages:
            perceptron = Perceptron()
            perceptron.language = language
            perceptron.train([data for data in training_data])
            self.perceptrons.append(perceptron)

    def test(self, test_data):
        correctly_predicted = 0
        total_files = 0

        for folder_name in os.listdir(test_data):
            folder_path = os.path.join(test_data, folder_name)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                letters_count = self.parse_file_to_dict(file_path)

                results = {perceptron.language: perceptron.predict(letters_count, False) for perceptron in self.perceptrons}
                print(results)

                predicted_language = max(results, key=results.get)
                if predicted_language == folder_name:
                    correctly_predicted += 1
                total_files += 1

        accuracy = (correctly_predicted / total_files) * 100 if total_files > 0 else 0
        print("Accuracy:", accuracy)

    def load_training_data(self, training_data_dir):
        training_data = []

        for language_dir in os.listdir(training_data_dir):
            language_path = os.path.join(training_data_dir, language_dir)
            for file_name in os.listdir(language_path):
                file_path = os.path.join(language_path, file_name)
                letters_count = self.parse_file_to_dict(file_path)
                training_data.append(letters_count)

        return training_data

    def parse_file_to_dict(self, file_path):
        letters_count = {letter: 0 for letter in string.ascii_lowercase}
        total_letters = 0

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                for letter in line:
                    lowercase_letter = letter.lower()
                    if lowercase_letter in letters_count:
                        total_letters += 1
                        letters_count[lowercase_letter] += 1

        for key in letters_count:
            if total_letters > 0:
                letters_count[key] = (letters_count[key] * 100) / total_letters

        path_parts = file_path.split(os.sep)
        language = path_parts[-2]
        letters_count['desired_output'] = language

        return letters_count


if __name__ == '__main__':
    neural_net = NeuralNetwork()
    training_data = neural_net.load_training_data("training_data")
    neural_net.train(training_data)
    neural_net.test("testing_data")
    neural_net.predict()
