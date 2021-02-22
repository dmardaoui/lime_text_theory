# -*- coding: utf-8 -*-

'''

Theory vs practice for a decision tree. In this script, we confront Proposition 3
to experimental observations. The result is Figure 6 
in the paper.

'''

# importations

from tqdm import tqdm
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pylab as plt
import matplotlib
from general.utils import extract
from general.plot import plot_whisker_boxes
from theory.beta import beta_f_decision_tree_three_words,beta_f_decision_tree_complex

# specific parameters
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 15})
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 



if __name__ == "__main__":

##############################Part 0: Parameters##############################

    # dataset
    path="dataset/positive_negative_reviews_yelp.csv"
    
    # classes of the study
    classes=['CLASS DISLIKE','CLASS LIKE']
    
    # number of samples
    n_samples=5000
    
    # number of features
    n_features=100
    
    # kernel width
    kernel_width=35
    
    # number of experiences
    n_exp=100
        
    # read dataset 
    df=pd.read_csv(path,sep='|') 
    X=list(df["text"])
    y=list(df["stars"])
    
    # split into a group of train and a group of test
    X_train, X_test, y_train, y_test= X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=11)    
    
    # sample: you can change the sample by simply changing the index
    sample=X_test[188] 
        
    # TF-IDF transformation
    vectorizer = TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)
    
##############################Part 1: Get the words##############################

    # random forest
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(train_vectors, y_train)
    
    # c is an object which performs the TF-IDF transformation and pass them through the model
    c = make_pipeline(vectorizer, rf)
    
    # creating the explainer 
    explainer = LimeTextExplainer(class_names=classes,kernel_width=kernel_width)
    
    # generation of an explanation
    exp = explainer.explain_instance(sample, c.predict_proba,num_samples=n_samples, num_features=n_features)
    
    # a list containing the words and associated beta
    res=exp.as_list(label=1)
    
    # extract the local dictionary
    local_dict=[]
    for i in np.linspace(0,len(res)-1,len(res)):
        inter=res[int(i)]
        w=inter[0] 
        local_dict.append(w)
    local_dict=sorted(local_dict)
    
    # words of the decision tree
    wl="food"
    wm="about"
    wt="Everything"
    
    # indices of the words in the local dictionary
    l=local_dict.index(wl)
    m=local_dict.index(wm)
    t=local_dict.index(wt)
    
    # indices of the words in the global dictionary
    global_dict=vectorizer.get_feature_names()
    index_l=global_dict.index(wl)
    index_m=global_dict.index(wm)
    index_t=global_dict.index(wt)
    
    
    # words of the decision tree
    '''wl="food"
    wm="about"
    wt="Everything"
    wr='bad'
    wv='character'
    
    #indices of the words in the local dictionary
    l=local_dict.index(wl)
    m=local_dict.index(wm)
    t=local_dict.index(wt)
    r=local_dict.index(wr)
    v=local_dict.index(wv)
    
    #indices of the words in the global dictionary
    global_dict=vectorizer.get_feature_names()
    index_l=global_dict.index(wl)
    index_m=global_dict.index(wm)
    index_t=global_dict.index(wt)
    index_r=global_dict.index(wr)
    index_v=global_dict.index(wv)'''
    
    
##############################Part 2: Make n_exp experiences of Lime ##############################  
    
    # function of the decision tree considering three words
    
    # decision_tree 1 = 1_{food} + (1-1_{food}) 1_{about} 1_{Everything}
    def func_decision_tree_simple(data):
        n_data = len(data)
        
        # matrix associated with the word l 
        res1 = np.zeros((n_data,2))
        
        # matrix associated with the word m
        res2 = np.zeros((n_data,2))
        
        # matrix associated with the word t
        res3 = np.zeros((n_data,2))
        
        # matrix associated with a combination of the words l,m,t
        res4 = np.zeros((n_data,2))
        
        # TF-IDF transformation of the sample of interest
        tfidf = vectorizer.transform(data)
        
        for i in range(n_data):
            
            # Indicator of word l 
            if tfidf[i,index_l] > 0.0:
                res1[i,1] = 1.0
            # Indicator of word m
            if tfidf[i,index_m] > 0.0:
                res2[i,1] = 1.0
            # Indicator of word t 
            if tfidf[i,index_t] > 0.0:
                res3[i,1] = 1.0
        
        # combination of the indicators of words l,m,t
        for i in range(n_data):
            res4[i,1]=res1[i,1]+(1-res1[i,1])*res2[i,1]*res3[i,1]
        
        res4[:,0]=(-1)*res4[:,1]

        return res4
    
    # decision_tree 2 = 1_{food} + (1-1_{food}) 1_{about} 1_{Everything} + 1_{bad} + 1_{bad} 1_{character}
    '''def func_five(data):
        n_data = len(data)
        res1 = np.zeros((n_data,2))
        res2 = np.zeros((n_data,2))
        res3 = np.zeros((n_data,2))
        res4 = np.zeros((n_data,2))
        res5 = np.zeros((n_data,2))
        res6 = np.zeros((n_data,2))
        
        tfidf = vectorizer.transform(data)
        
        for i in range(n_data):
            if tfidf[i,index_l] > 0.0:
                res1[i,1] = 1.0
            if tfidf[i,index_m] > 0.0:
                res2[i,1] = 1.0
            if tfidf[i,index_t] > 0.0:
                res3[i,1] = 1.0
            if tfidf[i,index_r] > 0.0:
                res4[i,1] = 1.0
            if tfidf[i,index_v] > 0.0:
                res5[i,1] = 1.0
                
        for i in range(n_data):
            res6[i,1]=res1[i,1]+(1-res1[i,1])*res2[i,1]*res3[i,1]+res4[i,1]+res4[i,1]*res5[i,1]
                
        res6[:,0]=res6[:,1]

        return res6'''
    
    # compute the values of each word after n_exp experiments
    data_store = np.zeros((n_exp,len(local_dict)+1)) 
    func=func_decision_tree_simple
    '''func=func_five'''
    
    # we are going to make n_exp experiences of LIME
    for i in tqdm(range(0,n_exp)):    
        # generate an explanation
        exp = explainer.explain_instance(sample,func,num_samples=n_samples,num_features=n_features,labels=[0,1])
        
        # extract the list of the class dislike
        liste_dislike=extract(exp.as_list(label=0),1) 
        
        # extract the list of the class like
        liste_like=extract(exp.as_list(label=1),1)
        
        # extract the words
        words_m=extract(exp.as_list(label=1),0) 
        
        # create a dataframe with the informations for the two classes
        result={'Dislike': liste_dislike, 'Like': liste_like} 
        df_coefficients=pd.DataFrame(result,index=words_m)
        
        # sort dataframe by alphabetic order of the words
        df_coefficients.sort_index(inplace=True)
        
        # take the values 
        data_store[i,1:]=df_coefficients["Like"].values
        
        
##############################Part 3: Plot figures ##############################  

    # figures
    fig, ax = plt.subplots(figsize=(15,10))
    theory=beta_f_decision_tree_three_words(d=len(local_dict),l=l,m=m,t=t,nu=0.35)[0:12]
    '''theory=beta_f_decision_tree_complex(d=len(local_dict),l=l,m=m,t=t,r=r,v=v,nu=0.25)[0:12]'''
    
    plot_whisker_boxes(data_store[:,0:12],
                       ax,
                       title=r"Interpretable coefficients for a decision tree",
                       xlabel=None,
                       theo=theory,
                       rotate=False,
                       feature_names=local_dict[0:12],
                       ylims=[-0.3,0.7],
                       color="red")
    
    # save figures
    s_name = "results/decision_tree"
    plt.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)
    
