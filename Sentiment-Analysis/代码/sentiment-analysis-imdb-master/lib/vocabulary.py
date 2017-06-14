import re
import codecs

class Vocabulary:
    def __init__(self, vocab_path='dataset/imdb.vocab'):
        self.vocab = dict()
        self.vocab_path = vocab_path

    def build(self):
        with codecs.open(self.vocab_path, 'r', 'UTF-8') as trainfile:
            words = [x.strip().rstrip('\n') for x in trainfile.readlines()]    #去标点符号+分段
            self.vocab = dict((c, i + 1) for i, c in enumerate(words))         #enumerate 遍历序列中的元素以及它们的下标；i是下标 c 是元素
        return self

    def size(self):
        return len(self.vocab)

    def tokenize(self, text):
        return [x.strip() for x in re.split('(\W+)', text) if x.strip()]  #分词

    def vectorize(self, text):
        text = text.lower()  #大写字母转成小写字母
        words = filter(lambda x: x in self.vocab, self.tokenize(text))
        return [self.vocab[w] for w in words]

