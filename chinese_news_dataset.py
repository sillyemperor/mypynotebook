import logging
from collections import Counter
from torchtext.vocab import Vocab
from tqdm import tqdm
import io
from torchtext.utils import unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
import torch


def _setup_datasets(root='.data', ngrams=2, vocab=None, include_unk=False):
    import os.path
    extracted_files = [os.path.join(root, 'train.csv'), os.path.join(root, 'test.csv')]

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    logging.info('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))


def build_vocab_from_iterator(iterator):
    """
    Build a Vocab from an iterator.

    Arguments:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
    """

    counter = Counter()
    with tqdm(unit_scale=0, unit='lines') as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    word_vocab = Vocab(counter)
    return word_vocab


def get_tokenizer(_):
    import jieba
    def tokenizer(s):
        return list(jieba.cut(s, cut_all=False))
    return tokenizer


def _csv_iterator(data_path, ngrams, yield_cls=False):
    tokenizer = get_tokenizer("jieba")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = ' '.join(row[1:])
            tokens = tokenizer(tokens)
            if yield_cls:
                yield int(row[0]) - 1, ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)


def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                        for token in tokens]))
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info('Row contains no tokens.')
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)


class TextClassificationDataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:

             - AG_NEWS
             - SogouNews
             - DBpedia
             - YelpReviewPolarity
             - YelpReviewFull
             - YahooAnswers
             - AmazonReviewPolarity
             - AmazonReviewFull

    """

    def __init__(self, vocab, data, labels):
        """Initiate text-classification dataset.

        Arguments:
            vocab: Vocabulary object used for dataset.
            data: a list of label/tokens tuple. tokens are a tensor after
                numericalizing the string tokens. label is an integer.
                [(label1, tokens1), (label2, tokens2), (label2, tokens3)]
            label: a set of the labels.
                {label1, label2}

        Examples:
            See the examples in examples/text_classification/

        """

        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab


def CN_NEWS(*args, **kwargs):
    """ Defines CN_NEWS datasets.
            The labels includes:
                1:fortune
                2:mil
                3:sport

        Arguments:
            root: Directory where the datasets are saved. Default: ".data"
            ngrams: a contiguous sequence of n items from s string text.
                Default: 1
            vocab: Vocabulary used for dataset. If None, it will generate a new
                vocabulary based on the train data set.
            include_unk: include unknown token in the data (Default: False)

        """
    return _setup_datasets(**kwargs)


if __name__ == '__main__':
    train_dataset, test_dataset = CN_NEWS(root='data/chinese_news')
    print(len(train_dataset.get_vocab()))
