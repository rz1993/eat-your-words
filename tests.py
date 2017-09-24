import unittest
import random

from models import HiddenMarkovModel as HMM
from probability import (FreqDist, ConditionalProbDist,
                         ConditionalFreqDist, MLEProbDist)

text = """I went to the store today so
        you can pretty much say I am a badass
        Some people say I am but Im really not you
        who tries too hard to be a badass though my
        brother""".split()
states = ['NOUN', 'VERB', 'PREPOSITION', 'ADJ']
tags = [random.choice(states) for w in text]
tagged_text = [list(zip(text, tags))]


class CustomTestCase(unittest.TestCase):
    """ Wrapper over unittest.TestCase for custom functionalities """
    def assertAlmostOne(self, number, **kwargs):
        # For rounding errors
        margin=10**4
        return self.assertTrue(abs(number-1) < margin,
            **kwargs)


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
            self.assertTrue(total,
                "Distribution for {} doesn't sum to one".format(tag))


class TestHMM(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.model = HMM()
        cls.model.fit(tagged_text)

    def test_hmm_train(self):
        model = self.model
        for attr in ('_states', '_priors', '_symbols'):
            self.assertTrue(getattr(model, attr))

        self.assertEquals(model._transition_matrix().shape,
                          (len(states), len(states)))
        for s in model._symbols:
            self.assertEquals(len(model._emission_vectors(s)), len(states))

    def test_hmm_emissions(self):
        model = self.model
        for w in text:
            print("Emission vector for {}".format(w))
            print(model._emission_vectors(w))

    def test_hmm_forward_prob(self):
        # Will result in -inf's since not the text is not vocab dense
        # i.e. there are only a few occurrences for each word so
        # since there are 4 tags, there will be a lot of unseen emissions
        model = self.model
        print("States: {}".format(" ".join(model._states)) if getattr(model, '_states') else "No States.")
        print("=========Testing Forward Algorithm=============")
        print(model.forward_prob(text))

    def test_viterbi_algorithm(self):
        model = self.model
        print(model.viterbi(text))


if __name__ == '__main__':
    unittest.main()
