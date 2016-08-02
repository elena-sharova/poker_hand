# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 15:29:09 2016

@author: Elena
"""
import collections as colls
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO 

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer

import numpy as np
import pandas 
import pydotplus
import graphviz
import urllib2

############ GLOBAL CONSTANTS ############
features=['FullHouse','Flush', 'FourOfAKind', 'ThreeOfAKind', 'TwoPairs', 'Straight','OnePair',
          'suit1','suit2','suit3','suit4','suit5',
           'rank1','rank2','rank3','rank4','rank5']
target = 'class'
pos = 1
neg = 0


############ FUNCTION DEFINITIONS ########
def read_data_into_dataframe(file_name, columns):
    return pandas.DataFrame(pandas.read_csv(filepath_or_buffer=file_name, 
                                    sep=',', header=0, dtype=int, names=columns))


# Used to determine if a hand consists of consecutive ranks
def consecutive_ranks(ranks):
    int_ranks = map(int, ranks)
    sorted_ranks = sorted(int_ranks)
    
    diffs = [x - sorted_ranks[i - 1] for i, x in enumerate(sorted_ranks) if i > 0]
    if max(diffs)==min(diffs)==1:
        return True
    else:
        # when 1 is among the cards, we need to treat it as an ace
        if (1 in sorted_ranks) and (13 in sorted_ranks) and (12 in sorted_ranks)             and (11 in sorted_ranks) and (10 in sorted_ranks):
            return True
        else:
            return False


# Used to concatenate all ranks into a string of ranks
def concatenate_all_ranks(savein, new_col_name):
    
    savein['rank1']=savein['r1'].apply(lambda x: str(x))
    savein['rank2']=savein['r2'].apply(lambda x: str(x))
    savein['rank3']=savein['r3'].apply(lambda x: str(x))
    savein['rank4']=savein['r4'].apply(lambda x: str(x))
    savein['rank5']=savein['r5'].apply(lambda x: str(x))
    
    savein[new_col_name]=savein['rank1']+','+savein['rank2']     +','+savein['rank3']+','+savein['rank4']+','+savein['rank5']


def sum_all_ranks(savein, new_col_name):
    
    savein[new_col_name]=savein['r1']+savein['r2']+savein['r3']+savein['r4']+savein['r5']


# Used to concatenate all suits into a string of suits
def concatenate_all_suits(savein, new_col_name):
    
    savein['suit1']=savein['s1'].apply(lambda x: str(x))
    savein['suit2']=savein['s2'].apply(lambda x: str(x))
    savein['suit3']=savein['s3'].apply(lambda x: str(x))
    savein['suit4']=savein['s4'].apply(lambda x: str(x))
    savein['suit5']=savein['s5'].apply(lambda x: str(x))
    
    savein[new_col_name]=savein['suit1']+savein['suit2']     +savein['suit3']+savein['suit4']+savein['suit5']


# Adding new high-level features
def add_new_features(savein, suits_col, ranks_col):
    
    savein['Flush']=savein[suits_col].apply(lambda x: pos if max(colls.Counter(x).values())==5 else neg)
    savein['FourOfAKind']=savein[ranks_col].apply(lambda x: pos if max(colls.Counter(x.split(",")).values())==4 else neg)
    savein['ThreeOfAKind']=savein[ranks_col].apply(lambda x: pos if max(colls.Counter(x.split(",")).values())==3 else neg)
    savein['TwoPairs']=savein[ranks_col].apply(lambda x: pos if (max(colls.Counter(x.split(",")).values())==2 
                                                                    and min(colls.Counter(x.split(",")).values())==1 
                                                                    and len(colls.Counter(x.split(",")).values())==3) 
                                                                    else neg)
    savein['Straight']=savein[ranks_col].apply(lambda x: pos if consecutive_ranks(colls.Counter(x.split(",")).keys())
                                                 else neg) 
    savein['OnePair']=savein[ranks_col].apply(lambda x: pos if (2 in colls.Counter(x.split(",")).values())
                                                                    else neg)
    savein['FullHouse']=savein['rank_sum'].apply(lambda x: pos if x==47 else neg)


def drop_unused_features(savein, feature_list):
    
    savein.drop(feature_list, 
              axis=1, inplace=True)


def add_class_specific_feature(c, class_name, savein):
    
    savein[class_name+str(c)]=savein[class_name].apply(lambda x: c if x==c else 0)


def fit_decision_tree(class_name, criter, fitdata, maxd):
    
    X=fitdata[features]
    Y = fitdata[class_name]
    model_to_fit=DecisionTreeClassifier(max_depth=maxd, criterion=criter)
    return model_to_fit.fit(X,Y)

def graph_decision_tree(model, class_names):
    
    model_dot = StringIO() 
    tree.export_graphviz(model, out_file=model_dot,
                         feature_names=features,
                         class_names=class_names,
                         filled=True, rounded=True,  
                         special_characters=True) 
    graph = pydotplus.graph_from_dot_data(model_dot.getvalue()) 
    graph.write_pdf("model"+class_names[1]+".pdf")

def graph_decision_tree(model, class_names):
    # Generate a pdf for each decision tree
    
    model_dot = StringIO() 
    tree.export_graphviz(model, out_file=model_dot,
                         feature_names=features,
                         class_names=class_names,
                         filled=True, rounded=True,  
                         special_characters=True) 
    graph = pydotplus.graph_from_dot_data(model_dot.getvalue()) 
    graph.write_pdf("model"+class_names[1]+".pdf")

if __name__ == '__main__':
      
           
      url = "http://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data"
      url2 = "http://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data"

      train_file = urllib2.urlopen(url)
      test_file = urllib2.urlopen(url2)
      # Read in the files into dataframes and add column names
      sf_train = read_data_into_dataframe(train_file,['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5','class'])
      sf_test = read_data_into_dataframe(test_file,['s1','r1','s2','r2','s3','r3','s4','r4','s5','r5','class'])
      
      concatenate_all_ranks(sf_train, 'all_ranks')
      concatenate_all_ranks(sf_test, 'all_ranks')
      
      concatenate_all_suits(sf_train, 'all_suits')
      concatenate_all_suits(sf_test, 'all_suits')
    
      sum_all_ranks(sf_train, 'rank_sum')
      sum_all_ranks(sf_test, 'rank_sum')

      add_new_features(sf_train, 'all_suits', 'all_ranks')
      add_new_features(sf_test, 'all_suits', 'all_ranks')

      drop_unused_features(sf_train,['all_ranks','all_suits','r1','r2','r3','r4','r5','s1','s2','s3','s4','s5', 'rank_sum'])
      drop_unused_features(sf_test,['all_ranks','all_suits','r1','r2','r3','r4','r5','s1','s2','s3','s4','s5', 'rank_sum'])

      for i in xrange(1,10):
          add_class_specific_feature(i, 'class', sf_train)

      modellist= []

      for i in xrange(1,10):
          modellist.append(fit_decision_tree('class'+str(i), 'entropy', sf_train, 5))
        
      test_X_01=sf_test[features]
      test_Y_01=sf_test[target]

      results = []

      for i in xrange(1,10):
          results.append(modellist[i-1].predict_proba(test_X_01)[:,1])


      # Optional - graph generated decision trees
      # Uncomment if would like to see the trees
      #for i in xrange(1,10):
      #    graph_decision_tree(modellist[i-1], ["0", str(i)])

      res1=1.0-results[0]
      n = res1.shape[0]
      predictions=n*[None]
      acc=0
      for i in xrange(n):
          max_prob = max([results[1][i],results[2][i],results[3][i],results[4][i],results[5][i],results[6][i],
                        results[7][i],results[8][i]])
          if max_prob>=0.9:
            predictions[i]=[results[1][i],results[2][i],results[3][i],results[4][i],results[5][i],results[6][i],
                        results[7][i],results[8][i]].index(max_prob)+2
          else:
              if res1[i]>results[0][i]:
                  predictions[i]=0
              else:
                  predictions[i]=1
        # compare prediction with actual class
        #print "Actual %s , predictions %s" %(test_Y_01[i], predictions[i])
        
          if test_Y_01[i]==predictions[i]:
              acc=acc+1
        #else:
            #print "Actual is %s,predicting %s" %(test_Y_01[i], predictions[i])
            #print i
      print "Achieved accuracy: %s" %(float(acc)/n)
