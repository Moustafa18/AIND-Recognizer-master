import warnings
from asl_data import SinglesData
import numpy as np

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    


    
    for each_word in range(0, len(test_set.get_all_Xlengths())):
        log_Likely_Hoods = {}
        current_sequences, current_sequences_length = test_set.get_item_Xlengths(each_word)
        #score_bic = 0
        # Calculate Log Likelyhood for each word 
        for word, model in models.items():
            try:
                score = model.score(current_sequences, current_sequences_length)
                log_Likely_Hoods[word] = score
                #num_Of_Data_Points = sum(model.lengths)
                    
                #num_Of_Params = 2 ** 2  + ( 2 * 2 * num_Of_Data_Points ) - 1 
                #score_bic = (-2 * log_Likely_Hood) + (num_Of_Params * np.log(num_Of_Data_Points))
            except:
                continue
            
        probabilities.append(log_Likely_Hoods)
        # compare index with the value
        guesses.append(max(log_Likely_Hoods, key = log_Likely_Hoods.get))


    return probabilities, guesses
