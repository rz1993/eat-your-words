from probability import ConditionalFreqDist as CFD, \
                        ConditionalProbDist as CPD, \
                        FreqDist, MLEProbDist
from util import logsumexp, sumexp

import errors
import numpy as np
import random

""" Constants """

_SEED = 1
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

    def train_unsupervised(self, unlabeled_seq, model, transformer=None):
        pass


class HiddenMarkovModelTagger:
    def __init__(self, transform=identity):
        self._transform = transform
        self._cache = None
        self._fitted = False

    def _create_cache(self):
        """
        Creates a cache for the HMM Tagger if one does not already
        exist; this allows for faster indexing and array operations
        for decoding hidden states and calculating sequence
        likelihoods.

        Returns:
            (tuple): Tuple of (T, E, P, sym_map) where:
                        T is a 2D numpy array of transition log probs
                        E is a 2D numpy array of emission log probs
                        P is a numpy array of prior log probs
                        sym_map is a reverse mapping of symbols to
                            integers, simplifies symbol lookups
        """

        if not self._cache:
            N, M = len(self._states), len(self._symbols)
            T = np.zeros((N, N), np.float64)
            E = np.zeros((N, M), np.float64)
            P = np.zeros(N, np.float32)
            sym_map = dict()
            states = self._states
            symbols = self._symbols

            for i in range(N):
                si = states[i]
                for j in range(N):
                    T[i, j] = self._transition_log_prob(si, states[j])
                for j in range(M):
                    E[i, j] = self._emission_log_prob(si, symbols[j])
                    P[i] = self._priors.log_prob(si)

            for i in range(M):
                sym_map[symbols[i]] = i

            self._cache = (T, E, P, sym_map)
            return self._cache

    def _update_cache(self, unlabeled_seq):
        """
        Updates the cache with new symbols given an unlabeled
        sequence input. The symbol vocabulary is expanded and
        the matrix of emission log probabilities is updated as
        well.
        """
        self._create_cache()
        M = len(self._symbols)
        N = len(self._states)
        T, E, P, sym_map = self._cache
        for s in unlabeled_seq:
            if s not in self._symbols:
                self._cache = None
                self._symbols.append(s)
        M_new = len(self._symbols)
        E = np.hstack((E, np.zeros((N, M_new-M), np.float64)))
        for i in range(M, M_new):
            si = self._symbols[i]
            sym_map[si] = i
            for j in range(N):
                E[j, i] = self._emission_log_prob(self._state[j], si)

        self._cache = (T, E, P, sym_map)

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
        self._create_cache()
        return self._cache[0]

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
        self._create_cache()
        T, E, P, sym_map = self._cache
        prev_prob = make_array((S,))
        prob = make_array((S,))

        for s in range(S):
            prev_prob[s] = P[s] + E[s, sym_map[unlabeled_seq[0]]]

        for i in range(1, t+1):
            symi = unlabeled_seq[i]
            for s in range(S):
                prob[s] = logsumexp(prev_prob + T[:, s]) \
                          + E[s, sym_map[symi]]
            prev_prob, prob = prob, prev_prob

        return prev_prob

    def _sample_prob(self, dist, prob, log=True):
        if log:
            dist = 2 ** dist
        total = 0
        for i in range(len(dist)):
            total_new = total + dist[i]
            if total <= prob <= total_new:
                return i
            total = total_new
        if total != 1:
            raise errors.InvalidProbDist()

    def _set_parameters(self, transitions, emissions, priors,
                        states, symbols):
        if self._fitted:
            self._update_parameters(transitions, emissions,
                                    priors, states, symbols)
        else:
            self._transitions = transitions
            self._emissions = emissions
            self._priors = priors
            self._states = states
            self._symbols = symbols
            self._fitted = True

    def _reconstruct_path(self, back_ptrs, index, t):
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
        sequence.reverse()
        return sequence

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
        return sumexp(self._forward_log_prob(unlabeled_seq, t))

    def forward_log_prob(self, unlabeled_seq, t=None):
        return math.log(self.forward_prob(unlabeled_seq, t))

    def backward_prob(self, unlabeled_seq, t):
        n = len(unlabeled_seq)
        unlabeled_seq.reverse()
        return self.forward_prob(unlabeled_seq, n-t)

    def backward_log_prob(self, unlabeled_seq, t):
        return math.log(self.backward_prob(unlabeled_seq, t))

    def best_path(self, unlabeled_seq, transform=None):
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
        if transform:
            unlabeled_seq = transform(unlabeled_seq)
        self._update_cache(unlabeled_seq)
        # HMM Parameter matrices from cache
        Tr, E, P, sym_map = self._cache
        T = len(unlabeled_seq)
        S = len(P)
        # Construct matrix of back pointers and arrays for storing
        # max probabilities for possible states for the previous
        # and current iteration
        back_ptrs = make_array((T, S), dtype=int)
        prev_maxes = make_array(S)
        maxes = make_array(S)
        for i in range(S):
            back_ptrs[0, i] = i
            prev_maxes[i] = P[i] + E[i, sym_map[unlabeled_seq[0]]]

        for j in range(1, T):
            symj = unlabeled_seq[j]
            for k in range(S):
                 probs = prev_maxes + Tr[:, k]
                 back_ptrs[j, k] = np.argmax(probs)
                 maxes[k] = probs[back_ptrs[j, k]] + E[k, sym_map[symj]]
            prev_maxes, maxes = maxes, prev_maxes

        return self._reconstruct_path(back_ptrs, np.argmax(prev_maxes), T)

    def random_sample(self, t, rng=None):
        """
        Generate a random sample based on Hidden Markov Model parameters
        Args:
            t (int): length of desired sample
            rng (object): any instance of a random generator that implements
                          a 'seed' method and a 'random' method which
                          returns a value between 0 and 1 inclusive.
        Returns:
            (list of length t): list of (state, output) tuples
        """
        sample = []
        self._create_cache()
        T, E, P, sym_map = self._cache
        if rng is None:
            rng = random
        rng.seed(_SEED)
        s0 = self._sample_prob(P, rng.random())
        o = self._sample_prob(E[s,:], rng.random())
        sample.append(self._states[s0], self._symbols[o])

        for i in range(1, t):
            s = self._sample_prob(T[s0,:], rng.random())
            o = self._sample_prob(E[s,:], rng.random())
            s0 = s
            sample.append(self_states[s], self._symbols[o])

        return sample
