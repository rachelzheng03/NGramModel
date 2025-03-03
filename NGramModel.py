from collections import defaultdict
from util import *
import statistics
import math

class NGramModel:
    def __init__(self, n = None, pretrained_data_file = None):
        if pretrained_data_file!=None:
            pretrained_data = load_model_data(pretrained_data_file)
            self.n = pretrained_data["n"]
            self.ngram_counts = {tuple(json.loads(k)): v for k, v in pretrained_data["ngram_counts"].items()}
            self.n_minus_1_gram_counts = {tuple(json.loads(k)): v for k, v in pretrained_data["n_minus_one_counts"].items()}
            self.vocab = set(pretrained_data["vocab"])

        else:
            self.n = n  # N-gram size
            self.ngram_counts = defaultdict(int)  # Counts of N-grams
            self.n_minus_1_gram_counts = defaultdict(int)  # Counts of (N-1)-grams
            self.vocab = set()  # Unique tokens in dataset

    def train(self, tokenized_methods: list):
        """
        Train the ngram model (ie. get the frequency of each possible ngram and (n-1)-gram)

        Args:
            tokenized_methods (list): A list of tokenized methods (represented by a list of tokens in the method)
        """
        train_size = len(tokenized_methods)
        for count, method in enumerate(tokenized_methods):
            tokens = ["<s>"] * (self.n - 1) + method + ["</s>"]  # Start & end tokens

            # extract all possible ngrams and (n-1)-grams from the method and update their counts (relative to the corpus)
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i : i + self.n])
                n_minus_1_gram = tuple(tokens[i : i + self.n - 1])
                
                self.ngram_counts[ngram] += 1
                self.n_minus_1_gram_counts[n_minus_1_gram] += 1
                self.vocab.update(tokens)

            if count%200 == 0:
                print_progress(int((count/train_size)*100))

        self.ngram_counts = dict(self.ngram_counts)
        self.n_minus_1_gram_counts = dict(self.n_minus_1_gram_counts)
        print_progress(100)
        print()

    def __get_probability(self, ngram:tuple):
        """
        Get the probability of an ngram (with laplace smoothing)

        Args:
            ngram: tuple of n tokens
        """

        n_minus_1 = ngram[:-1] 
     
        # laplace smoothing
        ngram_count = (self.ngram_counts[ngram] + 1) if ngram in self.ngram_counts.keys() else 1
        n_minus_1_count = (self.n_minus_1_gram_counts[n_minus_1] + len(self.vocab)) if n_minus_1 in self.n_minus_1_gram_counts.keys() else len(self.vocab)

        return ngram_count / n_minus_1_count
    
    def calc_perplexity(self, tokenized_method:list):
        """
        Compute perplexity of a given test sequence

        Args:
            tokenized_method: a list of tokens that make up a method
        """
        tokenized_method = ["<s>"] * (self.n - 1) + tokenized_method + ["</s>"]
        log_prob_sum = 0
        N = len(tokenized_method) - self.n + 1
        for i in range(N):
            ngram = tuple(tokenized_method[i : i + self.n])
            prob = self.__get_probability(ngram)
            log_prob_sum += math.log(prob)

        perplexity = math.exp(-log_prob_sum / N)
        return perplexity

    def evaluate_model(self, test_data: list):
        # calculate the perplexity of every method in test data
        perplexities = []
        for tokenized_method in test_data:
            perplexities.append(self.calc_perplexity(tokenized_method))

        # average the perplexities
        avg_perplexity = statistics.mean(perplexities)
        return avg_perplexity

    def generate_next_token(self, tokens: list):
        """
        Generate the next token given a line of code

        Args: 
            tokens: a list of tokens
        """
        padded_method = ["<s>"] * (self.n - 1) + tokens
        n_minus_1 = tuple(padded_method[-(self.n - 1):]) # the last n-1 tokens

        # candidates for the next token (find all ngrams that have the same n-1 tokens before it)
        candidates = [(ngram[-1], self.__get_probability(ngram)) for ngram in self.ngram_counts if ngram[:-1] == n_minus_1]
        
        if len(candidates) != 0:
            best_candidate = max(candidates, key=lambda x: x[1])  # Pick most probable token
            return best_candidate
        else:
            return ("<UNK>", 1/len(self.vocab))
    
    def generate_full_method(self, method_start, num_tokens_in_method = None, result = None):
        if result == None:
            result = []
        next_token = self.generate_next_token(method_start)

        # predicted token is end token
        if next_token[0] == "</s>":
            return result
        
        result.append(repr(next_token))
        if next_token[0] == "<UNK>":
            return result
    
        if len(method_start) + 1 == num_tokens_in_method:
            return result
        else:
            result = self.generate_full_method(method_start + [next_token[0]], num_tokens_in_method, result)
            return result
            
