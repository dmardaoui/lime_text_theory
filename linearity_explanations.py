# -*- coding: utf-8 -*-

'''

Linearity of the explanations. In this script, we confront Proposition 2 
to experimental observations. The result is Figure 5 
in the paper.

'''

# import
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
from tqdm import tqdm


# set parameters
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 15})
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 


if __name__ == "__main__":
    
############################# Part 0: Parameters #############################
    
    # data path
    path="dataset/positive_negative_reviews_yelp.csv"
    
    #force reproducibility
    np.random.seed(0)

    # classes of the mdoel 
    class_names=["Dislike","Like"]
    
    # number of samples
    n_samples=5000
    
    # number of features
    n_features=100
    
    # kernel width
    kernel_width=25
    
    # number of experiments
    n_exp=100

############################# Part 1: Process Data ###########################
    
    # read dataset
    df=pd.read_csv(path,sep='|') 
    X=list(df["text"])
    y=list(df["stars"])
    
    # split data into two groups
    X_train, X_test, y_train, y_test= X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=11)    
    
    # sample: you can change the sample by simply changing the index
    sample=X_test[188]
        
    # TF-IDF transformation
    vectorizer = TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)
    
############################# Part 2: Get the words ##########################
    
    # random forest 
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(train_vectors, y_train)
    
    # pipeline : TFIDF + Model
    c = make_pipeline(vectorizer, rf)

    # explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    #  generation of an explanation
    exp = explainer.explain_instance(sample, c.predict_proba,num_samples=n_samples, num_features=n_features)
    
    # table with the components of the explanation
    res=exp.as_list(label=1)
    
    # get the local dictionary
    local_dict=[]
    for i in np.linspace(0,len(res)-1,len(res)):
        inter=res[int(i)]
        w=inter[0] 
        local_dict.append(w)
    local_dict=sorted(local_dict)

############################# Part 3: Linearity ##########################    
    
    # random forest 1  
    rf1 = RandomForestClassifier(random_state=10)
    rf1.fit(train_vectors,y_train)
    
    # random forest 2 
    rf2 = RandomForestClassifier(random_state=15)
    rf2.fit(train_vectors,y_train)
    
    # random forest 3 
    rf3 = RandomForestClassifier(random_state=22)
    rf3.estimators_=rf1.estimators_+rf2.estimators_
    rf3.n_classes_=rf1.n_classes_
     
    # model 1 
    def model_rf1(data):
        n_data = len(data)
        res = np.zeros((n_data,2))
        tfidf = vectorizer.transform(data)
        p=rf1.predict_proba(tfidf)
        res[:,0] = p[:,1]
        res[:,1] = p[:,1]
        return res
    
    # model 2 
    def model_rf2(data):
        n_data = len(data)
        res = np.zeros((n_data,2))
        tfidf = vectorizer.transform(data)
        p=rf2.predict_proba(tfidf)
        res[:,0] = p[:,1]
        res[:,1] = p[:,1]
        return res
    
    
    # model 3 
    def model_sum(data):
        tfidf = vectorizer.transform(data)
        a1=rf1.predict_proba(tfidf)
        a2=rf2.predict_proba(tfidf)
        r=a1+a2
        return r

    
    # whisker box for the model 1 
    data_store_1 = np.zeros((n_exp,len(local_dict)+1)) 
    # we are going to make n_exp experiences of LIME
    for i in tqdm(range(0,n_exp)):
        explainer = LimeTextExplainer(class_names=class_names)
        exp_1 = explainer.explain_instance(sample,model_rf1,num_samples=n_samples,num_features=n_features,labels=[0,1])
        liste_dislike=extract(exp_1.as_list(label=0),1) 
        liste_like=extract(exp_1.as_list(label=1),1)
        words_m=extract(exp_1.as_list(label=1),0) 
        result={'Dislike': liste_dislike, 'Like': liste_like} 
        df_coefficients=pd.DataFrame(result,index=words_m)
        df_coefficients.sort_index(inplace=True)
        data_store_1[i,1:]=df_coefficients["Like"].values
    

        
   
    
    # whisker box for the model 2 
    data_store_2 = np.zeros((n_exp,len(local_dict)+1)) 
    # we are going to make n_exp experiences of LIME
    for i in tqdm(range(0,n_exp)):
        explainer = LimeTextExplainer(class_names=class_names)
        exp_2 = explainer.explain_instance(sample,model_rf2,num_samples=n_samples,num_features=n_features,labels=[0,1])
        liste_dislike=extract(exp_2.as_list(label=0),1) 
        liste_like=extract(exp_2.as_list(label=1),1)
        words_m=extract(exp_2.as_list(label=1),0) 
        result={'Dislike': liste_dislike, 'Like': liste_like} 
        df_coefficients=pd.DataFrame(result,index=words_m)
        df_coefficients.sort_index(inplace=True)
        data_store_2[i,1:]=df_coefficients["Like"].values


    # whisker box for the model 3 
    data_store_sum = np.zeros((n_exp,len(local_dict)+1)) 
    # we are going to make n_exp experiences of LIME
    for i in tqdm(range(0,n_exp)):
        explainer = LimeTextExplainer(class_names=class_names)
        exp_sum = explainer.explain_instance(sample,model_sum,num_samples=n_samples,num_features=n_features,labels=[0,1])
        liste_dislike=extract(exp_sum.as_list(label=0),1) 
        liste_like=extract(exp_sum.as_list(label=1),1)
        words_m=extract(exp_sum.as_list(label=1),0) 
        result={'Dislike': liste_dislike, 'Like': liste_like} 
        df_coefficients=pd.DataFrame(result,index=words_m)
        df_coefficients.sort_index(inplace=True)
        data_store_sum[i,1:]=df_coefficients["Like"].values
        

############################# Part 4: figures ##########################   
        
    # figure 1
    fig, ax = plt.subplots(figsize=(15,10))
    fig.suptitle("Linearity of explanations" , fontsize=40)
    
    bp1=plot_whisker_boxes(data_store_1[:,0:12],
                           ax,
                           title=r" ",
                           xlabel=None,
                           theo=None,
                           rotate=False,
                           feature_names=local_dict[0:11],
                           ylims=[round(-0.1,2),round(0.1,2)],
                           color="red",
                           c1='orange',
                           alpha=1,
                           c2="orange",
                           label="Model 1",
                           c3="orange")
    
    
    # figure 2
    bp2=plot_whisker_boxes(data_store_2[:,0:12],
                           ax,
                           title=r" ",
                           xlabel=None,
                           theo=None,
                           rotate=False,
                           feature_names=local_dict[0:11],
                           ylims=[round(-0.1,2),round(0.1,2)],
                           color="red",
                           c1='blue',
                           alpha=1,
                           c2="blue",
                           label="Model 2",
                           c3="blue")

    
    # figure 3
    bp3=plot_whisker_boxes(data_store_sum[:,0:12],
                           ax,
                           title=r"",
                           xlabel=None,
                           theo=None,
                           rotate=False,
                           feature_names=local_dict[0:11],
                           ylims=[round(-0.1,2),round(0.1,2)],
                           color="red",
                           c1="black",
                           c2='black',
                           label="Model 1 + Model 2",
                           c3="black")
    
    plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['Model 1', 'Model 2', 'Model 1 + Model 2'], loc='upper right',fontsize=28)
   
    # save figures
    s_name = "results/linsum"
    plt.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)
    
    
    
        
    

    
    
    
   
    
    
    
    
    
  