
# An Analysis of LIME for Text Data 


This is the code for the article ``An Analysis of LIME for Text Data.''(https://arxiv.org/abs/2010.12487)

There is nothing to install, you just have to run the scripts.


## Disclaimer 


The code was tested with version 0.2.0.0 of LIME, using another version may lead to unexplained behaviors. 


## General Organization 


The main scripts producing the results for the paper are in the root directory:

 * beta_in_function_of_nu: beta in function of the bandwidth
 * theory_meets_practice_decision_tree: checking our theoretical predictions for small decision tree
 * theory_meets_practice_linear: checking our theoretical predictions for linear models
 * linearity_explanations: demonstrating the linearity phenomenon

Useful functions are grouped into four folders:

* dataset/ Here you can find two datasets:
  * a dataset from Kaggle called Restaurants.ts: https://www.kaggle.com/hj5992/restaurantreviews
  * a subset of a dataset from Yelp Reviews: https://www.kaggle.com/omkarsabnis/yelp-reviews-dataset

* general/
  * plot.py: plot functions
  * utils.py: useful functions 

* results/ All the plots will end up here.

* theory/ contains all the functions needed for our theoretical predictions
 * alphas.py: computation of the alpha coefficients
 * sigmas.py: computation of the sigma coefficients
 * beta: computation of beta for some specific models



