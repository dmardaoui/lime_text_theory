# -*- coding: utf-8 -*-

'''

Evolution of beta coefficients in function of nu. The result is Figure 4 
in the paper.

'''

# importation
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pylab as plt
import matplotlib
from general.plot import plot_cancellation
from general.utils import extract
from tqdm import tqdm

# set parameters 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 15})



if __name__ == "__main__":

############################# Part 0: Parameters #############################
    
    # path of the dataset
    path="dataset/restaurants.tsv"
    
    # for reproducibility
    np.random.seed(10000)

    # default kernel width
    kernel_width=25
    
    # number of samples
    n_samples=100
    
    # number of features
    n_features=100
    
    # maximum bandwidth
    bandwidth_max=50
    
    # number of experiments
    n_exp=100
    
    # classes
    class_names=["Dislike","Like"]
    
    # class of interest
    class_interest="like"
    index_interest=1
    
############################ Part 1: Data Preparation ########################
    
    # read data 
    df=pd.read_csv(path,sep='\t') 
    
    # X
    X=list(df['Review'])
    
    # y
    y=list(df['Liked'])
    
    # split
    X_train, X_test, y_train, y_test= X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=10)
    
    # TF-IDF transformation
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)
    
    # model
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(train_vectors, y_train)
    
    # sample: you can change the sample by simply changing the index
    sample=X_test[5]
    
############################## Part 2: Get the words #########################
    
    # perform TF-IDF transformation +  Pass through the model
    c = make_pipeline(vectorizer, rf)
    
    # explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    # generate an explanation
    exp = explainer.explain_instance(sample, c.predict_proba,num_samples=n_samples, num_features=n_features)
    
    # explanation: beta and words 
    res=exp.as_list(label=1)
    
    # get the local dictionary
    local_dict=[]
    for i in np.linspace(0,len(res)-1,len(res)):
        inter=res[int(i)]
        w=inter[0]
        local_dict.append(w)
    local_dict=sorted(local_dict)
    
#################### Part 3: Lime after n_exp experiments ####################
    
    # length of the local dictionary
    n_beta=len(local_dict) 
    
    # matrix of beta 
    matrix_beta_kw=np.zeros((n_beta,bandwidth_max),dtype=float) 
    
    # matrix of variances
    matrix_variances=np.zeros((n_beta,bandwidth_max),dtype=float)
    
    # compute beta for different nu 
    for nu in tqdm(np.linspace(1,bandwidth_max,bandwidth_max)):
        
        # stock beta for n_exp experiments
        beta_x=np.zeros((n_beta,n_exp),dtype=float) 
        
        # we are going to make i_rep experiences of LIME for each nu 
        for irep in np.linspace(1,n_exp,n_exp):
                       
            # explanation object
            explainer = LimeTextExplainer(class_names=class_names,
                                          bow=True,
                                          feature_selection='none',
                                          kernel_width=nu) 
        
            # explanation
            exp = explainer.explain_instance(text_instance=sample,
                                             classifier_fn=c.predict_proba, 
                                             labels=[0,1],
                                             num_samples=n_samples,
                                             num_features=n_features,
                                             model_regressor=None) 
    
            # table of coefficients beta
            list_dislike=extract(exp.as_list(label=0),1) 
            list_like=extract(exp.as_list(label=1),1)
            
            # extract words
            words=extract(exp.as_list(label=1),0) 
            
            # combine the result in a data frame
            result={'Dislike': list_dislike, 'Like': list_like}
            df_coefficients=pd.DataFrame(result,index=words)
            df_coefficients=df_coefficients.sort_index() 
            
            # take the values of beta 
            B=df_coefficients.values
            
            # beta of the interest class
            beta_x[:,int(irep-1)]=B[:,index_interest]
            
        
        # compute the mean of beta after n_exp experiments
        mean_beta_exp=np.mean(beta_x, axis=1) 
  
        # put the mean of each beta in this matrix at the column nu corresponding
        matrix_beta_kw[:,int(nu-1)]=mean_beta_exp
        
        # compute the variance
        for std in np.linspace(0,n_beta-1,n_beta):
            variance_intermediaire=np.std(np.array(beta_x[int(std),:]))
            matrix_variances[int(std)][int(nu-1)]=variance_intermediaire
        
        
#################### Part 4: Plot figures ####################

#plot figures
plot_cancellation(n_beta=n_beta,
                  words=local_dict,
                  bandwidth_max=bandwidth_max,
                  matrix_beta_kw=matrix_beta_kw,
                  matrix_variances=matrix_variances)

    