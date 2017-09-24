from collections import Counter, defaultdict

import math
import numpy as np
import random
import statsmodels as stats

_N_INF = float('-1e300')

class FreqDist(Counter):
    def __init__(self, tokens=None):
        self._N = None
        super(FreqDist, self).__init__(tokens)

    def __setitem__(self, key, value):
        self._N = None
        super(FreqDist, self).__setitem__(key, value)

    def __delitem__(self, key, value):
        self._N = None
        super(FreqDist, self).__delitem__(key, value)

    def N(self):
        if self._N is None:
            self._N = float(sum(self.values()))
        return self._N

    def B(self):
        return len(self)

    def bins(self):
        return self.keys()

    def freq(self, key):
        n = self.N()
        return self[key] / n if n != 0 else 0

    def freq_counts(self, bins):
        counts = defaultdict(int)

        for count in self.values():
            counts[count] += 1

        return counts

    def max(self):
        best_v = 0
        best_k = None
        for (k, v) in self.items():
            if v > best_v:
                best_v = v
                best_k = k
        return best_k

    def _cumulative_freq(self, samples):
        cf = 0.0
        for sample in samples:
            cf += self[sample]
            yield cf


class ProbDist:
    #@abstractmethod
    def __init__(self):
        pass

    #@abstractmethod
    def prob(self, key):
        pass

    #@abstractmethod
    def vals(self):
        pass

    #@abstractmethod
    def max(self):
        pass

    def log_prob(self, key):
        prob = self.prob(key)
        return math.log(prob) if prob > 0 else _N_INF

    def generate(self):
        p = random.random()

        for k in self.vals():
            p -= self.prob(k)
            if p <= 0:
                return k

        # For some reason if there we don't return
        return random.choice(self.vals())


class MLEProbDist(ProbDist):
    def __init__(self, freq_dist):
        self._freq_dist = freq_dist

    def vals(self):
        return self._freq_dist.keys()

    def prob(self, sample):
        return self._freq_dist.freq(sample)

    def max(self):
        return self._freq_dist.max()

    def freq_dist(self):
        return self._freq_dist

    def total(self):
        total = 0.
        for k in self._freq_dist:
            total += self.prob(k)
        return total

    def __repr__(self):
        return "<MLEProbDist with {} keys>".format(len(self.vals()))


class ConditionalFreqDist(defaultdict):
    def __init__(self, labeled_seq=None):
        super(ConditionalFreqDist, self).__init__(FreqDist)

        if labeled_seq is not None:
            for (cond, token) in labeled_seq:
                self[cond][token] += 1


class ConditionalProbDist(dict):
    def __init__(self, cfd, prob_dist_factory,
                       smoother=lambda x: x+1):
        self._smoother = smoother
        self._prob_dist_factory = prob_dist_factory
        # Add possible args and kwargs for prod_dist_factory
        for c in cfd:
            self[c] = prob_dist_factory(cfd.get(c))

    def conditions(self):
        return self.keys()

    def __missing__(self, key):
        self[key] = self._prob_dist_factory(FreqDist())
        return self[key]

    def __repr__(self):
        return "<Condition Probability Distribution with {} conditions>".format(len(self.conditions()))
