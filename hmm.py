# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:10:03 2018

@author: WIN
"""

import numpy as np
import pandas as pd
from asl_data import AslDb
from asl_utils import test_features_tryit

asl = AslDb() # initializes the database
s  = asl.df.head() # displays the first five rows of the asl database, indexed by video and frame


asl.df.iloc[99,1]  # look at the data available for an individual frame

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']

asl.df.head()  # the new feature 'grnd-ry' is now in the frames dictionary


# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences 
# between hand and nose locations

asl.df['grnd-rx'] =  asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] =  asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] =  asl.df['left-x'] - asl.df['nose-x']

# test the code
test_features_tryit(asl)

# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
 #show a single set of features for a given (video, frame) tuple
print([asl.df.ix[98,1][v] for v in features_ground])

