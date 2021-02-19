# -*- coding: utf-8 -*-

'''

Theory vs practice for a linear model. In this script, we confront Proposition 4 to 
experimental observations. The result is Figure 7 
in the paper.

'''

# import 
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
from theory.beta import beta_f_linear

# set parameters
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 15})
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 

if __name__ == "__main__":
    
############################# Part 0: Parameters #############################
    
    # for reproducibility
    np.random.seed(30)

    # classes of the model 
    class_names=["Dislike","Like"] 
    
    # number of samples
    n_samples=5000
    
    # number of features
    n_features=1000
    
    # bandwidth 
    kernel_width=25
    
    # number of experiments 
    n_exp=100
    
    # data path
    path="dataset/positive_negative_reviews_yelp.csv"
    
    # class of interest 
    class_interest="Like"
    
############################# Part 1: Process Data ###########################
    
    # read data 
    df=pd.read_csv(path,sep='|') 
    X=list(df["text"])
    y=list(df["stars"])
    
    # split into a group of train and test 
    X_train, X_test, y_train, y_test= X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=11)    
        
    # sample: you can change the sample by simply changing the index
    sample=X_test[13]
    #sample=X_test[20]
    
    # TF-IDF transformation 
    vectorizer = TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)
    
############################# Part 3: Get the words ##########################
    
    # model
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(train_vectors, y_train)
    
    # pipeline: Vectorizer + Model
    c = make_pipeline(vectorizer, rf)
    
    # create an explanation
    explainer = LimeTextExplainer(class_names=class_names,kernel_width=kernel_width)
    
    # generate an explanation 
    exp = explainer.explain_instance(sample, c.predict_proba,num_samples=n_samples, num_features=n_features)
    
    # table with the informations: beta + words 
    res=exp.as_list(label=1)
    
    # get the local dictionary
    local_dict=[]
    for i in np.linspace(0,len(res)-1,len(res)):
        inter=res[int(i)]
        w=inter[0]  
        local_dict.append(w)
    local_dict=sorted(local_dict)
    
    # get the global dictionary
    global_dict=vectorizer.get_feature_names()
    size_dico = test_vectors.shape[1]
    
    # get the words which are in the local dictionary and in the global dictionary
    subset_words=[]
    indices_subset=[]
    for i,word in enumerate(local_dict):
        if word in global_dict :
            subset_words.append(word)
            indices_subset.append(i)

    size_subset=len(subset_words)
    
############################# Part 4: Linear function#########################
    
    # random numbers 
    np.random.seed(30) 
    vec = np.random.normal(0,1,(size_dico,))

    # tfidf transformation of the sample
    tfidf = vectorizer.transform([sample])
    
    # linear function 
    def linear_function(data):
        n_data = len(data)
        res = np.zeros((n_data,2))
        tfidf = vectorizer.transform(data)
        res[:,0] = tfidf * vec
        res[:,1] = tfidf * vec
        return res

    
    # compute values of each coefficient after n_exp experiences
    data_store = np.zeros((n_exp,size_subset+1)) 

    # we are going to make n_exp experiences of LIME
    for i in tqdm(range(0,n_exp)):
        
        # generation of an explanation 
        exp = explainer.explain_instance(sample,linear_function,num_samples=n_samples,num_features=n_features,labels=[0,1])
        
        # list containing the coefficients of the class dislike
        liste_dislike=extract(exp.as_list(label=0),1) 
        
        # list containing the coefficients of the class like 
        liste_like=extract(exp.as_list(label=1),1)
        
        # words of the sample 
        words_m=extract(exp.as_list(label=1),0) 
        
        # table with the summary of the explanation 
        result={'Dislike': liste_dislike, 'Like': liste_like} 
        df_coefficients=pd.DataFrame(result,index=words_m)
        df_coefficients.sort_index(inplace=True)
        
        values=df_coefficients[class_interest].values
        
        # take the class of interest 
        for n,index in enumerate(indices_subset):
            data_store[i,n+1]=values[index]
        
################################## Part 5: figures ##########################

    fig, ax = plt.subplots(figsize=(15,10))
    
    plot_whisker_boxes(data_store[:,30:60],
                       ax,
                       title="Interpretable coefficients for a linear function",
                       xlabel=None,
                       theo=beta_f_linear(d=size_subset,tfidf=tfidf,f=vec,dico=subset_words,gd=global_dict)[30:60],
                       rotate=False,
                       feature_names=subset_words[30:60],
                       ylims=[-1,1],
                       color="red")

    
    # save figure
    s_name = "results/linear"
    plt.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)
