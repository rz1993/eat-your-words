from probability import ConditionalFreqDist as CFD, \
                        ConditionalProbDist as CPD, \
                        FreqDist, MLEProbDist

import errors
import numpy as np

""" Constants """

_TEXT = 0
_TAG = 1

""" Utility functions """

def identity(seq):
    return seq

def make_array(shape, dtype=np.float64):
    arr = np.zeros(shape, dtype)
    if dtype==np.float64:
        arr.fill(-np.inf)
    return arr

def logexpsum(np_arr):
    np_arr = 2**(np_arr)
    result = np.log2(np.sum(np_arr))
    return result

""" HMM Model """

class HiddenMarkovModelTrainer:
    def __init__(self, states, symbols):
        self._states = states
        self._symbols = symbols

    def train_supervised(self, seq, model):
        """
        Fits the HMM parameters, namely transition, emission and prior
        probabilities, while also initializing the list of states and
        emissions. Does this by using MLE estimates for all probabilities
        based on the labeled sequence that is passed to this method.
        Parameter estimates are then saved to the inputted model instance,
        via its '_set_parameters()' method.

        Args:
            labeled_seq (list): list of lists of (state, emission) tuple pairs
            model (HiddenMarkovModelTagger): the model instance to train
        """

        # Unpack the sentences and separate them into a tags
        # sequence (tags will be used to initialize priors)
        tags = [pair[_TAG] for sent in seq for pair in sent]

        transitions = CFD()
        emissions = CFD()
        states = set()
        symbols = set()

        # Train the conditional distributions by iterating through the
        # pairs and counting (state, emission) and (state_i, state_i+1)
        for sent in seq:
            n = len(sent)
            for i, pair in enumerate(sent):
                state, symbol = pair[_TAG], pair[_TEXT]
                if i < n-1:
                    transitions[state][sent[i+1][_TAG]] += 1
                emissions[state][symbol] += 1
                states.add(state)
                symbols.add(symbol)

        # Save the trained parameters to the model instance and wrap the
        # conditional frequencies with the ConditionalProbDist class
        model._set_parameters(transitions=CPD(transitions, MLEProbDist),
                              emissions=CPD(emissions, MLEProbDist),
                              priors=MLEProbDist(FreqDist(tags)),
                              states=list(states),
                              symbols=list(symbols))

    def train_unsupervised(self, unlabled_seq, transformer=None):
        pass


class HiddenMarkovModelTagger:
    def __init__(self, transform=identity):
        self._transform = transform
        self._cache = dict()
        self._fitted = False

    def _emission_log_prob(self, state, symbol):
        return self._emissions[state].log_prob(symbol)

    def _emission_vectors(self, symbol):
        return np.array([self._emission_log_prob(s, symbol)
                            for s in self._states])

    def _transition_log_prob(self, state1, state2):
        return self._transitions[state1].log_prob(state2)

    def _transition_matrix(self):
        """
        Returns:
            numpy.array (SxS): the transition probabilities in Matrix
                               form, for easier indexing and operations
        """
        if not 'T' in self._cache:
            self._cache['T'] = np.array([
                [self._transition_log_prob(s1, s2) for s2 in self._states]
                    for s1 in self._states
                ])
        return self._cache['T']

    def _forward_log_prob(self, unlabeled_seq, t=None):
        """
        Forward algorithm for calculating the probability
        of a sequence of tokens.

        Args:
            unlabeled_seq (list): list of items/emissions
            t (int): index of the emission to calculate probability for

        Returns:
            numpy.array (1xS): an array of the marginalized probabilities
            for all states at time = t.
        """

        T = len(unlabeled_seq)
        if t is None:
            t = T-1
        S = len(self._states)
        transitions = self._transition_matrix()
        prev_prob = make_array((S,))
        prob = make_array((S,))

        for s, state in enumerate(self._states):
            prev_prob[s] = self._priors.log_prob(state) + \
                          self._emission_log_prob(state, unlabeled_seq[0][_TEXT])

        for i in range(1, t+1):
            symbol = unlabeled_seq[i][_TEXT]
            for s, state in enumerate(self._states):
                prob[s] = logexpsum(prev_prob + transitions[:, s]) \
                          + self._emission_log_prob(state, symbol)
            prev_prob, prob = prob, prev_prob

        return prev_prob

    def _set_parameters(self, transitions, emissions, priors,
                        states, symbols):
        self._transitions = transitions
        self._emissions = emissions
        self._priors = priors
        self._states = states
        self._symbols = symbols
        self._fitted = True

    def train(self, labeled_seq, unlabeled_seq=None, **kwargs):
        """
        Called by an instance of the HiddenMarkovModelTagger.
        Fits the HMM parameters, namely transition, emission and prior
        probabilities, which are set by the HiddenMarkovModelTrainer
        class, which encapsulates supervised and unsupervised methods
        for training HMMs. A labeled sequence must be passed so that
        the trainer can be initialized with a state space.

        Args:
            labeled_seq (list): list of lists of (state, emission)
                             tuple pairs or just emissions
        """
        state_space = set(pair[_TAG] for sent in labeled_seq \
            for pair in sent)
        symbol_space = set(pair[_TEXT] for sent in labeled_seq \
            for pair in sent)

        trainer = HiddenMarkovModelTrainer(state_space, symbol_space)

        trainer.train_supervised(labeled_seq, model=self)
        if unlabeled_seq:
            max_iterations = kwargs.get('max_iterations', 10)
            trainer.train_unsupervised(unlabeled_seq, model=self,
                                       max_iterations=max_iterations)

    def forward_prob(self, unlabeled_seq, t=None):
        return np.sum(2**self._forward_log_prob(unlabeled_seq, t))

    def forward_log_prob(self, unlabeled_seq, t=None):
        return math.log(self.forward_prob(unlabeled_seq, t))

    def backward_prob(self, unlabeled_seq, t):
        n = len(unlabeled_seq)
        return self.forward_prob(unlabeled_seq[::-1], n-t)

    def backward_log_prob(self, unlabeled_seq, t):
        n = len(unlabled_seq)
        return self.forward_log_prob(unlabeled_seq[::-1], n-t)

    def best_state_path(self, back_ptrs, index, t):
        """
        Reconstruct the best path based on a matrix of back pointers
        and the optimal index in the last time t.

        Args:
            back_ptrs (np.array((TxS))): array of backward indices
            index (int): the optimal state index to navigate back from
        Returns:
            list: a list of states
        """

        sequence = [self._states[index]]
        for i in range(t-1, 0, -1):
            row = back_ptrs[i]
            index = row[index]
            sequence.append(self._states[index])
        return sequence[::-1]

    def viterbi(self, unlabeled_seq, t=None):
        """
        Find the state sequence with the highest likelihood for
        a sequence of unlabeled emissions, using a dynamic
        programming approach known as the Viterbi Algorithm, which
        takes advantage of the Markov property, that a specific state
        at time t only depends on a finite number of earlier states.

        Args:
            unlabeled_seq (list): a list of emissions without labels
        Returns:
            list: the best state path based on probability
        """

        T = len(unlabeled_seq)
        if t is None:
            t = T
        S = len(self._states)
        # Construct matrices for faster column indexing
        trans = self._transition_matrix()
        # Matrix of back pointers
        back_ptrs = make_array((t, S), dtype=int)

        prev_maxes = make_array(S)
        for i, s in enumerate(self._states):
            back_ptrs[0, i] = i
            prev_maxes[i] = self._priors.log_prob(s) + \
                            self._emission_log_prob(s, unlabeled_seq[0])

        maxes = make_array(S)
        for j in range(1, t):
            symbol = unlabeled_seq[j]
            for k, s in enumerate(self._states):
                 probs = prev_maxes + trans[:, k]
                 back_ptrs[j, k] = np.argmax(probs)
                 maxes[k] = probs[back_ptrs[j, k]] + \
                            self._emission_log_prob(s, symbol)
            prev_maxes, maxes = maxes, prev_maxes

        return self.best_state_path(back_ptrs, np.argmax(maxes), t)
