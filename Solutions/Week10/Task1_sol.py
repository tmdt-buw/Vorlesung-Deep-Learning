from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import random
import time
import math

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def findFiles(path):
    return glob.glob(path)


def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    """
    Read a file and split into lines
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def letterToIndex(letter):
    """
    Find letter index from all_letters, e.g. "a" = 0
    """
    return all_letters.find(letter)


def letterToTensor(letter):
    """
    Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    """
    Turn a line into a <line_length x 1 x n_letters>,
    or an array of one-hot letter vectors
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# ToDo:     Go to the RNN Class and implement the structure of the RNN.
# ToDo:     A schematic of the Network can be found here: https://i.imgur.com/Z2xbySO.png
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def train_net(self, category_tensor, line_tensor):
        hidden = self.initHidden()

        self.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in self.parameters():
            p.data.add_(-learning_rate, p.grad.data)

        return output, loss.item()

    # Just return an output given a line
    def evaluate(self, line_tensor):
        hidden = self.initHidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)

        return output

    def predict(self, input_line, n_predictions=3):
        print('\n> %s' % input_line)
        with torch.no_grad():
            output = self.evaluate(lineToTensor(input_line))

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, all_categories[category_index]))
                predictions.append([value, all_categories[category_index]])


if __name__ == "__main__":

    flags = {"train": True}

    # ToDo      Task 1a) Download the data from [1] and extract it to the current directory.
    # ToDo               [1] https://download.pytorch.org/tutorial/data.zip

    """
    Included in the data/names directory are 18 text files named as “[Language].txt”. Each file contains a bunch of names,
    one name per line, mostly romanized (but we still need to convert from Unicode to ASCII). We’ll end up with a dictionary
    of lists of names per language, {language: [names ...]}. The generic variables “category” and “line” 
    (for language and name in our case) are used for later extensibility.
    """
    print(findFiles('data/names/*.txt'))
    

    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    print(unicodeToAscii('Ślusàrski'))
    


    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)


    """
    Now we have category_lines, a dictionary mapping each category (language) to a list of lines (names).
    We also kept track of all_categories (just a list of languages) and n_categories for later reference.
    """
    print(category_lines['Italian'][:5])
    


    """
    Turning Names into Tensors
    Now that we have all the names organized, we need to turn them into Tensors to make any use of them.
    
    To represent a single letter, we use a “one-hot vector” of size <1 x n_letters>.
    A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.
    
    To make a word we join a bunch of those into a 2D matrix <line_length x 1 x n_letters>.
    
    That extra 1 dimension is because PyTorch assumes everything is in batches - we’re just using a batch size of 1 here.
    """
    print(letterToTensor('J'))
    print(lineToTensor('Jones').size())
    


    """
    Creating the Network
    Before autograd, creating a recurrent neural network in Torch involved cloning the parameters of a layer over several
    timesteps. The layers held hidden state and gradients which are now entirely handled by the graph itself.
    This means you can implement a RNN in a very “pure” way, as regular feed-forward layers.
    
    This RNN module (mostly copied from the PyTorch for Torch users tutorial) is just 2 linear layers which operate on an
    input and hidden state, with a LogSoftmax layer after the output.
    """
    n_hidden = 128
    # ToDo:     Go to the RNN Class and implement the structure of the RNN.
    # ToDo:     A schematic of the Network can be found here: https://i.imgur.com/Z2xbySO.png
    rnn = RNN(n_letters, n_hidden, n_categories)
    print(rnn)
    

    """ To run a step of this network we need to pass an input (in our case, the Tensor for the current letter) and a\
    previous hidden state (which we initialize as zeros at first). We’ll get back the output (probability of each language)
    and a next hidden state (which we keep for the next step).
    """
    input = letterToTensor('A')
    hidden = torch.zeros(1, n_hidden)
    output, next_hidden = rnn(input, hidden)
    print(output)
    

    """
    For the sake of efficiency we don’t want to be creating a new Tensor for every step, so we will use lineToTensor instead
    of letterToTensor and use slices. This could be further optimized by pre-computing batches of Tensors.
    """
    input = lineToTensor('Albert')
    hidden = torch.zeros(1, n_hidden)

    output, next_hidden = rnn(input[0], hidden)
    print(output)
    

    """
    As you can see the output is a <1 x n_categories> Tensor, where every item is the likelihood of that category
    (higher is more likely).
    """


    """
    Training
    Preparing for Training
    Before going into training we should make a few helper functions. The first is to interpret the output of the network,
    which we know to be a likelihood of each category. We can use Tensor.topk to get the index of the greatest value:
    """
    print(categoryFromOutput(output))
    


    """
    We will also want a quick way to get a training example (a name and its language):
    """
    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line)


    """
    Training the Network
    Now all it takes to train this network is show it a bunch of examples, have it make guesses, and tell it if it’s wrong.
    
    For the loss function nn.NLLLoss is appropriate, since the last layer of the RNN is nn.LogSoftmax.
    """
    # ToDo:     Chose an appropriate loss function. Remember the activation of the output layer of your network
    # ToDo:     and combine it with a loss function suitable for the classification problem.
    criterion = nn.NLLLoss()


    """
    Each loop of training will:
    
        Create input and target tensors
        Create a zeroed initial hidden state
        Read each letter in and
        Keep hidden state for next letter
        Compare final output to target
        Back-propagate
        Return the output and loss
    """

    # ToDo:     Find a good value for the learning rate. If you set this too high, it might explode.
    # ToDo:     If too low, it might not learn.
    learning_rate = 0.005


    """
    Now we just have to run that with a bunch of examples. Since the train function returns both the output and loss we can
    print its guesses and also keep track of loss for plotting. Since there are 1000s of examples we print only every
    print_every examples, and take an average of the loss.
    """
    # ToDo:     Chose an apropriate number of training iterations. Remember that too few iterations will stop the
    # ToDo:     training too early, too many iterations may lead to overfitting. Investigate your choice for the
    # ToDo:     parameters of the learning rate and the number of training iterations with provided plotting functions.

    # train
    if flags["train"]:
        n_iters = 200000
        print_every = 5000
        plot_every = 1000

        # Keep track of losses for plotting
        current_loss = 0
        all_losses = []

        start = time.time()
        for iter in range(1, n_iters + 1):
            category, line, category_tensor, line_tensor = randomTrainingExample()
            output, loss = rnn.train_net(category_tensor, line_tensor)
            current_loss += loss

            # Print iter number, loss, name and guess
            if iter % print_every == 0:
                guess, guess_i = categoryFromOutput(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

            # Add current loss avg to list of losses
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

        torch.save(rnn.state_dict(), f"rnn.pt")

        """
        Plotting the Results
        Plotting the historical loss from all_losses shows the network learning:
        """
        plt.figure()
        plt.plot(all_losses)
    
    if not flags["train"]:
        rnn.load_state_dict(torch.load(f"rnn.pt"))
        rnn.eval()

    """
    Evaluating the Results
    To see how well the network performs on different categories, we will create a confusion matrix, indicating for every
    actual language (rows) which language the network guesses (columns). To calculate the confusion matrix a bunch of
    samples are run through the network with evaluate(), which is the same as train() minus the backprop.
    """
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000


    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = rnn.evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

    """
    You can pick out bright spots off the main axis that show which languages it guesses incorrectly, e.g. Chinese for
    Korean, and Spanish for Italian. It seems to do very well with Greek, and very poorly with English (perhaps because
    of overlap with other languages).
    """


    """
    Running on User Input
    """
    # ToDo:     Use the network to predict the language of your own name.
    rnn.predict('Richard')
    rnn.predict('Fabian')
    rnn.predict('Max')
    rnn.predict('Marijke')
    rnn.predict('Pim')
    rnn.predict('Rüdiger')
    rnn.predict('Ruediger')