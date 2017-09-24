from probability import ConditionalFreqDist as CFD


class Token:
    def __init__(self, text, types, line):
        self.text = text
        self.types = types
        self.line = line

class ContextIndex(defaultdict):
    pass


class SequentialTagger:
    def __init__(self, default=None):
        if default is None:
            self._taggers = [self]
        else:
            self._taggers = [self] + default._taggers

    @property
    def backoff(self):
        return self._taggers[1] if len(self._taggers) > 1 else None

    def tag(self, tokens):
        tags = []
        for index in range(len(tokens)):
            tags.append(self.tag_token(tokens, index, tags))
        return list(zip(tokens, tags))

    def tag_token(self, tokens, index, history):
        tag = None
        for tagger in self._taggers:
            tag = tagger.choose_tag(tokens, index, history)
            if tag is not None:
                break
        return tag

    #@abstractmethod
    def choose_tag(self, tokens, index, history):
        pass


class ContextTagger(SequentialTagger):
    def __init__(self, context_to_tag=None):
        if context_to_tag is None:
            self.context_to_tag = ContextIndex()
        else:
            self.context_to_tag = context_to_tag
        super(ContextTagger, self).__init__()

    #@abstractmethod
    def context(self, tokens, index, history):
        pass

    def choose_tag(self, tokens, index, history):
        context = self.context(tokens, index, history)
        tag = self.context_to_tag.get(context)
        return tag

    def _train(self, tagged_text, cutoff=0, verbose=False):

        fd = CFD()

        useful_contexts = set()

        for sentence in tagged_text:
            tokens, tags = zip(*sentence)
            for i, (token, tag) in enumerate(sentence):
                context = self.context(tokens, i, tags[:i])
                if context is None:
                    continue
                fd[context][tag] += 1
                if (self.backoff is None or
                        tag != self.tag_token(
                        tokens, i, tags[:i])):
                    useful_contexts.add(context)

        for c in useful_contexts:
            best_tag = fd[c].max()
            hits = fd[c][best_tag]
            if hits > cutoff:
                self.context_to_tag[c] = best_tag


class NGramTagger(ContextTagger):
    def __init__(self, n, context_to_tag=None):
        self._n = n
        super(NGramTagger, self).__init__(context_to_word)

    def context(self, tokens, index, history):
        n = self._n
        if n > index:
            context = tuple(["START"] + history[:index])
        else:
            context = tuple(history[index-n:index])
        return context


class HiddenMarkovModelTagger:
    def __init__(self, states, transitions, emissions, priors):
        self._transitions = transitions
        self._emissions = emissions
        self._priors = priors
        self._states = states

    @classmethod
    def _train(cls, labeled_seq, test_seq=None,
                    estimator=None, **kwargs):
