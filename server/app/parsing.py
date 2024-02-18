# 20230117 fk: whole file taken from Felix HuBaFeedRL parsing.py (with own adjustments)

import nltk
import numpy as np
from functools import reduce

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob


class SimpleParser:
    '''
    SimpleParser class used to evaluate an input text based on contained keywords.
    '''

    def __init__(self) -> None:
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe('spacytextblob')
        self.scores = []
        self.stopw = nltk.corpus.stopwords.words('english')

    def segmentAnalysis(self, data: list) -> list:

        '''
        SimpleParser.segmentAnalysis() transforms an input text into proper
        segments, based on a sequence provided by the user. If none was given
        the input text is counted as single segment. After having segmented the
        input, a sentiment analysis for each segment will be computed based on
        predefined sentiment dictionaries.
        '''

        print(data)

        if data == []:
            # data input object is empty -> return 0 value
            return [0.0]
        else:
            # using a user defined segmentation
            for d in data:
                # preprocessing of the input segment
                doc = self.nlp(d)
                self.scores.append(doc._.blob.polarity)
        return self.scores

    def flatten(self, mode: str = None) -> float:

        '''
        Compresses multiple score values to a single score of type float.
            Works with two modes:
            > "avg" returns the average of the values
            > no mode returns the sum of all values
        '''

        if mode is not None:
            return np.mean(np.array(self.scores))
        return reduce(lambda a, b: a + b, self.scores)
