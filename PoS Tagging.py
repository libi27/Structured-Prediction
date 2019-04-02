import pickle
import numpy as np
import sys
from scipy.misc import logsumexp
import collections
import time
import itertools


START_STATE = '*START*'
START_WORD = '$START$'
END_STATE = '*END*'
END_WORD = '$END$'
RARE_WORD = '$RARE_WORD$'

INDEX_POS = 0
INDEX_SENTENCE = 1
RARE_THR = 3

def data_example(data_path='PoS_data.pickle',
                 words_path='all_words.pickle',
                 pos_path='all_PoS.pickle'):
    """
    An example function for loading and printing the Parts-of-Speech data for
    this exercise.
    Note that these do not contain the "rare" values and you will need to
    insert them yourself.

    :param data_path: the path of the PoS_data file.
    :param words_path: the path of the all_words file.
    :param pos_path: the path of the all_PoS file.
    """

    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    print("The number of sentences in the data set is: " + str(len(data)))
    print("\nThe tenth sentence in the data set, along with its PoS is:")
    print(data[10][1])
    print(data[10][0])

    print("\nThe number of words in the data set is: " + str(len(words)))
    print("The number of parts of speech in the data set is: " + str(len(pos)))

    print("one of the words is: " + words[34467])
    print("one of the parts of speech is: " + pos[17])

    print(pos)


class Baseline(object):
    '''
    The baseline model.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.e = np.zeros((self.pos_size, self.words_size)) + sys.float_info.min
        self.q = np.zeros(self.pos_size) + sys.float_info.min
        self.words_counter = collections.Counter([all_w for sentence in [d[INDEX_SENTENCE] for d in training_set] for all_w in sentence])

    def MAP(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''

        pos_tags = []
        for sentence in sentences:
            pos_index = np.argmax(self.e[:, [self.word2i[x if self.words_counter[x] > RARE_THR else RARE_WORD] for x in sentence]].T + self.q, axis=1)
            pos_tags.append([self.pos_tags[x] for x in pos_index])
        return pos_tags


def baseline_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    multinomial and emission probabilities for the baseline model.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial baseline model with the pos2i and word2i mappings among other things.
    :return: a mapping of the multinomial and emission probabilities. You may implement
            the probabilities in |PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """
    # Multinomial & Emission Table update
    for sequence in training_set:
        for current_pos, word in zip(sequence[0], sequence[1]):
            if model.words_counter[word] < RARE_THR:
                model.e[model.pos2i[current_pos]][model.word2i[RARE_WORD]] += 1
            else:
                model.e[model.pos2i[current_pos]][model.word2i[word]] += 1
            model.q[model.pos2i[current_pos]] += 1

    model.e = np.log((model.e.T / model.e.sum(axis=1)).T)
    model.q = np.log(model.q / model.q.sum())


class HMM(object):
    '''
    The basic HMM_Model with multinomial transition functions.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words)
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.t = np.zeros((self.pos_size, self.pos_size)) + sys.float_info.min
        self.e = np.zeros((self.pos_size, self.words_size)) + sys.float_info.min
        self.words_counter = collections.Counter([all_w for sentence in [d[INDEX_SENTENCE] for d in training_set] for all_w in sentence])

    def sample(self, n):
        '''
        Sample n sequences of words from the HMM.
        :return: A list of word sequences.
        '''
        sequences = []
        for i in range(n):
            sentence = [np.random.choice(self.words, 1, p=np.exp(self.e[self.pos2i[START_STATE], :]))[0]]
            current_pos = START_STATE
            while sentence[-1] != END_WORD:
                current_pos = np.random.choice(self.pos_tags, 1, p=np.exp(self.t[self.pos2i[current_pos], :]))[0]
                sentence.append(
                    np.random.choice(self.words, 1, p=np.exp(self.e[self.pos2i[current_pos], :]))[0])
            sequences.append(sentence)
        return sequences

    def viterbi(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''

        pos_sequences = []

        for sentence in sentences:
            sentence_length = len(sentence)
            v = np.zeros([sentence_length, self.pos_size])
            tracking_table = np.zeros([sentence_length, self.pos_size], dtype=int)

            # init step
            start_prob = np.zeros(self.pos_size) + sys.float_info.min
            start_prob[self.pos2i[START_STATE]] = 1
            v[0, :] = np.log(start_prob/np.sum(start_prob))
            tracking_table[0, :] = np.argmax(v[0, :])

            for i in range(1, sentence_length):
                w = sentence[i] if self.words_counter[sentence[i]] > RARE_THR else RARE_WORD
                # w = sentence[i]
                viterbi_step = v[i - 1, :] + (self.t + self.e[:, self.word2i[w]]).T
                tracking_table[i, :] = np.argmax(viterbi_step, axis=1)
                v[i, :] = np.diag(viterbi_step[:, tracking_table[i, :]])

            # Inference calculation from the tracking table built dynamically
            inference = np.zeros(sentence_length, dtype=int)
            inference[sentence_length - 1] = np.argmax(v[sentence_length - 1, :])
            for i in range(sentence_length - 2, -1, -1):
                inference[i] = tracking_table[i + 1, inference[i + 1]]
            pos_sequences.append([self.pos_tags[inference[i]] for i in range(sentence_length)])
            # return [self.pos_tags[inference[i]] for i in range(sentence_length)]
        return pos_sequences


def hmm_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    transition and emission probabilities for the standard multinomial HMM.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial HMM with the pos2i and word2i mappings among other things.
    :return: a mapping of the transition and emission probabilities. You may implement
            the probabilities in |PoS|x|PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """

    # Multinomial & Emission Table update
    for sequence in training_set:
        for current_pos, word in zip(sequence[INDEX_POS], sequence[INDEX_SENTENCE]):
            if model.words_counter[word] < RARE_THR:
                model.e[model.pos2i[current_pos]][model.word2i[RARE_WORD]] += 1
            else:
                model.e[model.pos2i[current_pos]][model.word2i[word]] += 1
        for pos1, pos2 in zip(sequence[INDEX_POS][:-1], sequence[INDEX_POS][1:]):
            model.t[model.pos2i[pos1]][model.pos2i[pos2]] += 1

    model.t = np.log((model.t.T / model.t.sum(axis=1)).T)
    model.e = np.log((model.e.T / model.e.sum(axis=1)).T)



class MEMM(object):
    '''
    The base Maximum Entropy Markov Model with log-linear transition functions.
    '''

    def __init__(self, pos_tags, words, phi, training_set):
        '''
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        :param phi: the feature mapping function, which accepts two PoS tags
                    and a word, and returns a list of indices that have a "1" in
                    the binary feature vector.
        '''
        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.words_counter = collections.Counter([all_w for sentence in [d[INDEX_SENTENCE] for d in training_set] for all_w in sentence])
        # self.phi = phi(self.pos_tags, self.words, self.pos2i, self.word2i, self.words_counter)
        self.phi, self.phi_size = phi(self.pos_tags, self.words, self.pos2i, self.word2i)

    def viterbi(self, sentences, w):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequences.
        '''

        pos_sequences = []

        for sentence in sentences:
            sentence_length = len(sentence)
            v = np.zeros([sentence_length, self.pos_size])
            tracking_table = np.zeros([sentence_length, self.pos_size], dtype=int)

            # init step
            start_prob = np.zeros(self.pos_size) + sys.float_info.min
            start_prob[self.pos2i[START_STATE]] = 1
            v[0, :] = np.log(start_prob / np.sum(start_prob))
            tracking_table[0, :] = np.argmax(v[0, :])

            for t in range(1, sentence_length):
                phi_normalize = []
                for pos_from in self.pos_tags:
                    phi_z = [
                        np.sum(w[self.phi[pos_to, pos_from] + self.phi[pos_to, sentence[t]]])
                        for pos_to in self.pos_tags]
                    res = logsumexp(phi_z)
                    phi_normalize.append(res)

                for ind, p in enumerate(self.pos_tags):
                    word = sentence[t]
                    viterbi_step = [
                        np.sum(w[self.phi[p, p_1] + self.phi[p, word]])-phi_normalize[p_1_i]
                        for p_1_i, p_1 in enumerate(self.pos_tags)
                        ]
                    v_to_max = np.array(viterbi_step) + v[t - 1, :]
                    v[t, ind] = np.max(v_to_max)
                    tracking_table[t, ind] = np.argmax(v_to_max)

            # Inference calculation from the tracking table built dynamically
            inference = np.zeros(sentence_length, dtype=int)
            inference[sentence_length - 1] = np.argmax(v[sentence_length - 1, :])
            for i in range(sentence_length - 2, -1, -1):
                inference[i] = tracking_table[i + 1, inference[i + 1]]
            pos_sequences.append([self.pos_tags[inference[i]] for i in range(sentence_length)])
            # return [self.pos_tags[inference[i]] for i in range(sentence_length)]

        return pos_sequences


def perceptron(training_set, initial_model, w0="", eta=0.1, epochs=1):
    """
    learn the weight vector of a log-linear model according to the training set.
    :param training_set: iterable sequence of sentences and their parts-of-speech.
    :param initial_model: an initial MEMM object, containing among other things
            the phi feature mapping function.
    :param w0: an initial weights vector.
    :param eta: the learning rate for the perceptron algorithm.
    :param epochs: the amount of times to go over the entire training data (default is 1).
    :return: w, the learned weights vector for the MEMM.
    """
    if type(w0) is np.ndarray:
        w = w0
    else:
        w = np.zeros(initial_model.phi_size)
    # sum_w = np.zeros(initial_model.phi_size)
    # w_tracking = []
    for epoch in range(epochs):

        for ind, train_pair in enumerate(training_set):
            x = train_pair[INDEX_SENTENCE]
            y = train_pair[INDEX_POS]
            y_predict = initial_model.viterbi([x], w)[0]
            np.add.at(w,
                      [val for sublist in [initial_model.phi[y[t], y[t - 1]]+initial_model.phi[y[t], x[t]] for t in range(1, len(x))] for val in
                       sublist], eta)
            np.add.at(w, [val for sublist in
                          [initial_model.phi[y_predict[t], y_predict[t - 1]]+initial_model.phi[y_predict[t], x[t]] for t in range(1, len(x))] for val in
                          sublist], -eta)

            np.add.at(w, list(itertools.chain.from_iterable(
                [initial_model.phi[y[t], y[t - 1]] + initial_model.phi[y[t], x[t]]
                 for t in range(1, len(x))])), eta)
            np.add.at(w, list(itertools.chain.from_iterable(
                [initial_model.phi[y_predict[t], y_predict[t - 1]] + initial_model.phi[y_predict[t], x[t]]
                 for t in range(1, len(x))])), -eta)

        #     sum_w += w
        # w_tracking.append(sum_w / len(training_set))
    # return w_tracking[-1]
    return w


def HMMFeature(pos_tags, words, pos2i, words2i):
    pos_size = len(pos2i)
    words_size = len(words2i)
    e_block_start = pos_size ** 2
    dict_t = {(pos1, pos2): [pos2i[pos2] * pos_size + pos2i[pos1]]
              for pos1 in pos_tags for pos2 in pos_tags}
    dict_e = {(pos2, word): [e_block_start + pos2i[pos2] * words_size + words2i[word]]
              for pos2 in pos_tags for word in words}

    return {**dict_t, **dict_e}, pos_size ** 2 + pos_size * words_size


def process_data(data_to_process, pos, words):
    for sentence in data_to_process:
        sentence[INDEX_POS].insert(0, START_STATE)
        sentence[INDEX_POS].append(END_STATE)

        sentence[INDEX_SENTENCE].insert(0, START_WORD)
        sentence[INDEX_SENTENCE].append(END_WORD)

    pos.extend([START_STATE, END_STATE])
    words.extend([START_WORD, END_WORD, RARE_WORD])


def save_model(filename, weights):
    f = open(filename, "wb")
    np.save(f, weights)
    f.close()


def load_model(filename):
    f = open(filename, "rb")
    weights = np.load(f)
    f.close()
    return weights


def test_model(testing_set, models, w):
    x_test = [test[INDEX_SENTENCE] for test in testing_set]
    y_test = [test[INDEX_POS] for test in testing_set]
    for model in models:
        score = 0
        if model.__class__.__name__ == "Baseline":
            predict_pos = model.MAP(x_test)
        elif model.__class__.__name__ == "HMM":
            predict_pos = model.viterbi(x_test)
        else:
            predict_pos = model.viterbi(x_test, w)
        for yreal, yhat in zip(y_test, predict_pos):
            score += ((np.array(yreal) == np.array(yhat)).sum()/len(yreal))
        print(model.__class__.__name__ + " Success Rate: " + str(score / len(x_test)))


if __name__ == '__main__':

    with open('PoS_data.pickle', 'rb') as file:
        data = pickle.load(file)
    with open('all_words.pickle', 'rb') as file:
        all_words = pickle.load(file)
    with open('all_PoS.pickle', 'rb') as file:
        all_pos = pickle.load(file)

    process_data(data, all_pos, all_words)
    print(sys.argv)
    if len(sys.argv) > 1:
        percent_learning = float(sys.argv[1])
        percent_testing = float(sys.argv[2])
        eta_ = float(sys.argv[3])
        epochs_ = int(sys.argv[4])
        name = sys.argv[5]
        train_set = data[int(np.round(len(data) * percent_learning)):int(np.round(len(data) * (percent_learning+0.1)))]
        test_set = data[int(np.round(len(data) * percent_testing)):]
        memm = MEMM(all_pos, all_words, HMMFeature, train_set)
        # w_init = load_model("MEMM_weights_10")
        start = time.time()
        w_learned = perceptron(train_set, memm, "", eta_, epochs_)
        end = time.time()
        print("MEMM Training Time for {}% of the data is {}".format(percent_learning * 100, str(end - start)))
        save_model(name, w_learned)
        test_model(test_set, [memm], w_learned)
    else:
        for percent in [0.1, 0.25, 0.9]:

            train_set = data[0:int(np.round(len(data) * percent))]
            test_set = data[int(np.round(len(data) * percent)):]
            baseline = Baseline(all_pos, all_words, train_set)
            start = time.time()
            baseline_mle(train_set, baseline)
            end = time.time()
            print("Baseline Training Time for {}% of the data is {}".format(percent*100,str(end-start)))
            hmm = HMM(all_pos, all_words, train_set)
            start = time.time()
            hmm_mle(train_set, hmm)
            end = time.time()
            print("HMM Training Time for {}% of the data is {}".format(percent*100, str(end - start)))

            test_model(test_set, [baseline, hmm], "")
        print(hmm.sample(5))

