import numpy as np

class Corpus(object):
    def data_processing(self, text):
        text = text.lower()
        text = text.replace('.', ' .')
        words = text.split(' ')

        word_to_id = {}
        id_to_word = {}

        for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word

        corpus = [word_to_id[word] for word in words]
        corpus = np.array(corpus)
        return corpus, word_to_id, id_to_word


    def creat_co_matrix(self, corpus, vocab_size, window_size = 1):
        corpus_size = len(corpus)
        co_matrix = np.zeros((corpus_size, vocab_size), dtype=np.int32)
        for id, word_id in enumerate(corpus):
            for i in range(1, window_size + 1):
                left_id = id - i
                right_id = id + i

                if left_id >= 0:
                    left_word_id = corpus[left_id]
                    co_matrix[word_id, left_word_id] += 1

                if right_id < corpus_size:
                    right_word_id = corpus[right_id]
                    co_matrix[word_id, right_word_id] += 1
        return co_matrix

    def cos_similarity(self, x, y, epsilon = 1e-8):
        nx = x / np.sqrt(np.sum(x**2) + epsilon)
        ny = y / np.sqrt(np.sum(y**2) + epsilon)
        return np.dot(nx, ny)


test = Corpus()
text = 'You say goodbye and I say Hello.'
corpus, word_to_id, id_to_word = test.data_processing(text)
vocab_size = len(word_to_id)
matrix = test.creat_co_matrix(corpus, vocab_size)
c1 = matrix[word_to_id['you']]
c2 = matrix[word_to_id['i']]
print(test.cos_similarity(c1, c2))
