import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        scores = []
        
        for num_Of_States in range(self.min_n_components, self.max_n_components + 1):
                try:
                   
                    
                    model = self.base_model(num_Of_States)
                    
                    log_Likely_Hood = model.score(self.X, self.lengths)
                    
                    num_Of_Data_Points = model.n_features
                    
                    num_Of_Params = num_Of_States ** 2  +  2 * num_Of_States * num_Of_Data_Points  - 1 
                    
                   
                    score_bic = -2 * log_Likely_Hood + num_Of_Params * np.log(len(self.X))
                    
                    scores.append([score_bic, model])
                except:
                    pass
          
        if len(scores) > 0:   
            best_score = float("Inf") 
            best_model = None 
            for count in range(1,len(scores)):
                    if scores[count][0] < best_score:
                        best_score = scores[count][0]
                        best_model = scores[count][1]
            return  best_model   

        else:
            return self.base_model(self.n_constant)
           
                
                
    

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float("-Inf")
        best_model = None

        try:
           
            model = None
            for num_Of_States in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(num_Of_States)
                scores = []
                for word, _model in self.hwords.items():
                    if word != self.this_word:
                        scores.append(model.score(_model))
                new_score = model.score(self.X, self.lengths) - np.mean(scores)
                
                if best_score < new_score:
                    best_score = new_score
                    best_model = model
                    #self.X, self.lengths = combine_sequences(range(len(self.sequences)), self.sequences)
                    #model = self.base_model(num_Of_States)
                    #best_score = model.score(self.X, self.lengths) - np.mean(scores)           
            return best_score
        except:
            return self.base_model(self.n_constant)       
        
        return best_score


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        split_method = KFold(n_splits=2)
        
        
        
        scores = []
        
        for num_Of_States in range(self.min_n_components, self.max_n_components + 1):
            log_Likely_Hoods = []
            try:
                if len(self.sequences) >= 3:
                    
                    
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        #print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
        
                    
                        
                        # split the index to get the training data
                        self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                        
                        # split the index to get the test data
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        
                        # fit the training data Actually I make it by calling base_model
                        model = self.base_model(num_Of_States)
                   
                        # calc the log likliy hood
                        log_Likely_Hood= model.score(X_test, lengths_test)
                else:
                    model = self.base_model(num_Of_States)
                    log_Likely_Hood = model.score(self.X, self.lengths)
                    
                log_Likely_Hoods.append(log_Likely_Hood)    
                # calculate the avg
                score_avg = np.mean(log_Likely_Hoods)
                scores.append([score_avg, model])
            except:
                return self.base_model(self.n_constant)
                
        #print(len(scores))         
        
        if len(scores) > 0:
            _max = scores[0][0]
            best_model = scores[0][1]
            for count in range(1,len(scores)):
                if _max < scores[count][0]:
                    _max = scores[count][0]
                    best_model = scores[count][1]
        else:
            return self.base_model(self.n_constant)
        
        return best_model
                 
                 
                
   