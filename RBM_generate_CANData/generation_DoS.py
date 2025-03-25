from __future__ import print_function
import numpy as np
import pandas as pd
import csv


class RBM:

    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.debug_print = True

        np_rng = np.random.RandomState(1234)   

        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))

        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

    def train(self, data, max_epochs=1000, learning_rate=0.1):
        """
        Train the machine.
        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the states of visible units.
        """

        num_examples = data.shape[0]

        data = np.insert(data, 0, 1, axis=1)

        for epoch in range(max_epochs):
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_probs[:, 0] = 1  # Fix the bias unit.
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            pos_associations = np.dot(data.T, pos_hidden_probs)

            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:, 0] = 1  # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Update weights.
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

            error = np.sum((data - neg_visible_probs) ** 2)
            if self.debug_print:
                print("Epoch %s: error is %s" % (epoch, error))

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible units.

        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        data = np.insert(data, 0, 1, axis=1)

        hidden_activations = np.dot(data, self.weights)

        hidden_probs = self._logistic(hidden_activations)

        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)

        hidden_states = hidden_states[:, 1:]
        return hidden_states

    def run_hidden(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of hidden units, to get a sample of the visible units.
        Parameters
        ----------
        data: A matrix where each row consists of the states of the hidden units.
        Returns
        -------
        visible_states: A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        visible_states = np.ones((num_examples, self.num_visible + 1))

        data = np.insert(data, 0, 1, axis=1)

        visible_activations = np.dot(data, self.weights.T)
        visible_probs = self._logistic(visible_activations)
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)

        visible_states = visible_states[:, 1:]
        return visible_states

    def daydream(self, num_samples):
        """
        Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
        (where each step consists of updating all the hidden units, and then updating all of the visible units),
        taking a sample of the visible units at each step.
        Note that we only initialize the network *once*, so these samples are correlated.
        Returns
        -------
        samples: A matrix, where each row is a sample of the visible units produced while the network was
        daydreaming.
        """
        samples = np.ones((num_samples, self.num_visible + 1))


        samples[0, 1:] = np.random.rand(self.num_visible)

        for i in range(1, num_samples):
            visible = samples[i - 1, :]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights)
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i, :] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:, 1:]

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    n_a=16
    n_b=10
    r = RBM(num_visible=n_a, num_hidden=n_b)
    choose_lines = 10000
    data = pd.read_csv("./car_hacking/csv/chuli/gai_dos_ttt_2.csv",
                       error_bad_lines=False, nrows=choose_lines)

    training_data = np.array(data)
    r.train(training_data, max_epochs=5000)
    print(r.weights)

    nn=10000    
    zz = np.zeros((nn, n_b), int)
    for i in range(0,nn):
        for j in range(0,n_b):
            zz[i][j]=np.random.choice([0,1])

    gen_data=np.zeros((nn, n_a), int)

    gen_data = r.run_hidden(zz)


    with open('./gen_data_rbm/another_gai_gen_dos_ttt.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        lll_list = np.arange(0, n_a, 1)
        writer.writerow(lll_list)
        writer.writerows(gen_data)

    file.close()