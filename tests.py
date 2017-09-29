import unittest
import numpy as np
import random

from models import HiddenMarkovModelTagger as HMM
from probability import (FreqDist, ConditionalProbDist,
                         ConditionalFreqDist, MLEProbDist)
from util import logsumexp, sumexp


text = """I went to the store today so
        you can pretty much say I am a badass
        Some people say I am but Im really not you
        who tries too hard to be a badass though my
        brother""".split()
states = ['NOUN', 'VERB', 'PREPOSITION', 'ADJ']
vocab = list(set(text))
tags = [random.choice(states) for w in text]
tagged_text = [list(zip(text, tags))]


class CustomTestCase(unittest.TestCase):
    """ Wrapper over unittest.TestCase for custom functionalities """
    def assertAlmostOne(self, number, **kwargs):
        # For rounding errors
        margin=10**-4
        error = kwargs.get('error', '{} is not close enough to one.'.format(number))
        return self.assertTrue(abs(number-1) < margin, error)


class TestProbability(CustomTestCase):
    @classmethod
    def setUp(cls):
        cls.tokens = text
        cond_text = list(zip(tags, text))
        cls.fd = FreqDist(tags)
        cls.cfd = ConditionalFreqDist(cond_text)
        cls.cpd = ConditionalProbDist(cls.cfd, MLEProbDist)

    def test_freq_dist(self):
        for tag in tags:
            total = self.cpd[tag].total()
            self.assertAlmostOne(total,
                error="Distribution for {} doesn't sum to one".format(tag))


class TestHMM(CustomTestCase):
    @classmethod
    def setUp(cls):
        cls.model = HMM()
        cls.model.train(tagged_text)

    def test_hmm_cache(self):
        model = self.model
        model._create_cache()
        T, E, P, sym_map = model._cache
        self.assertAlmostOne(sumexp(P))
        self.assertEqual(set(sym_map.keys()),
                         set(model._symbols))
        for s in range(len(P)):
            self.assertAlmostOne(sumexp(T[s, :]))
            self.assertAlmostOne(sumexp(E[s, :]))

    def test_hmm_parameters_initialized(self):
        model = self.model
        for attr in ('_states', '_priors', '_symbols'):
            self.assertTrue(getattr(model, attr))

    def test_hmm_transitions(self):
        model = self.model
        matrix = model._transition_matrix()
        self.assertEqual(matrix.shape, (len(states), len(states)))

        for i, s in enumerate(states):
            trans_row = model._transitions[s]
            self.assertAlmostOne(trans_row.total())
            self.assertAlmostOne(sumexp(matrix[i,:]))

    def test_hmm_emissions(self):
        model = self.model
        for s in model._symbols:
            self.assertEqual(len(model._emission_vectors(s)), len(states))

        for i, s in enumerate(states):
            emiss_row = model._emissions[s]
            filtered = [t[0] for t in tagged_text[0] if t[1] == s]
            self.assertAlmostOne(emiss_row.total())
            self.assertEqual(emiss_row.freq_dist().N(), len(filtered))

    def test_hmm_forward_prob(self):
        # Will result in -inf's since not the text is not vocab dense
        # i.e. there are only a few occurrences for each word so
        # since there are 4 tags, there will be a lot of unseen emissions
        model = self.model
        print("States: {}".format(" ".join(model._states)) if getattr(model, '_states') else "No States.")
        print("=========Testing Forward Algorithm=============")
        print(model.forward_prob(text))

    def test_best_path_algorithm(self):
        model = self.model
        print(model.best_path(text))

    def test_hmm_random_sample(self):
        model = self.model
        t = 10
        #print(model.random_sample(t))


if __name__ == '__main__':
    unittest.main()
