import collections
import numpy as np
import re

def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in xrange(len(sample)-n+1):
                yield sample[i:i+n]

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        # p is another NgramLanguageModel
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        # p is another NgramLanguageModel
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i**2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i**2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        # p is another NgramLanguageModel
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            # num += np.exp(p.log_likelihood(ngram))
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    # def js_with(self, p):
    #     # KL(P||M)
    #     log_likelihood_ratios = []
    #     for ngram in p.ngrams():
    #         m_log_likelihood = np.log(0.5*np.exp(p.log_likelihood(ngram)) + 0.5*np.exp(self.log_likelihood(ngram)))
    #         log_likelihood_ratios.append(p.log_likelihood(ngram) - m_log_likelihood)
    #     kl_p_m = np.mean(log_likelihood_ratios)

    #     # KL(Q||M)
    #     log_likelihood_ratios = []
    #     for ngram in self.ngrams():
    #         m_log_likelihood = np.log(0.5*np.exp(p.log_likelihood(ngram)) + 0.5*np.exp(self.log_likelihood(ngram)))
    #         log_likelihood_ratios.append(self.log_likelihood(ngram) - m_log_likelihood)
    #     kl_q_m = np.mean(log_likelihood_ratios)

    #     return np.mean([kl_p_m, kl_q_m])
    def js_with(self, p):
        return self.fast_js_with(p)

    def fast_js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5*(kl_p_m + kl_q_m) / np.log(2)

def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048):
    lines = []

    # print "WARNING SHAKESPEARE"
    # with open('/media/ramdisk/shakespeare.txt', 'r') as f:
    #     for line in f:
    #         line = line[:-1]
    #         if len(line) > 16:
    #             line = line[:16]
    #         lines.append(line + ("`"*(16-len(line))))

    finished = False

    # print 'WARNING OPENSUBTITLES'
    # for i in xrange(2316):
    #     path = "/media/ramdisk/opensubtitles-parser/data/{}raw.txt".format(str(i+1))
    for i in xrange(99):
        path = "/home/ishaan/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-{}-of-00100".format(str(i+1).zfill(5))
        with open(path, 'r') as f:
            for line in f:
                line = line[:-1]
                if tokenize:
                    line = tokenize_string(line)
                else:
                    line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ("`",)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in xrange(100):
        print filtered_lines[i]

    print "loaded {} lines in dataset".format(len(lines))
    return filtered_lines, charmap, inv_charmap

def load_dataset_big(batch_size, max_length):
    all_lines = []
    for i in xrange(99):
        path = "/home/ishaan/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-{}-of-00100".format(str(i+1).zfill(5))
        with open(path, 'r') as f:
            for line in f:
                line = line[:-1]+' ' # Replace trailing \n with a space
                all_lines.append(line)

    charmap = {}
    inv_charmap = []

    for line in all_lines:
        for char in line:
            if char not in charmap:
                charmap[char] = len(inv_charmap)
                inv_charmap.append(char)

    def get_epoch():
        np.random.shuffle(all_lines)
        all_lines_idx = 0
        buffer = ''
        while True:
            while len(buffer) < max_length*batch_size:
                if all_lines_idx >= len(all_lines):
                    return
                buffer += all_lines[all_lines_idx]
                all_lines_idx += 1
            yield [buffer[max_length*i:max_length*(i+1)] 
                for i in xrange(batch_size)]
            buffer = buffer[batch_size*max_length:]

    return get_epoch, charmap, inv_charmap