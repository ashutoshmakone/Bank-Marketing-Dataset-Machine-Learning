# Bank-Marketing-Dataset-Machine-Learning
The classification goal is to predict if the client will subscribe (yes/no) a term deposit. EDA followed by modeling with KNN, NB, LR, LR with Polynomial Features, SVM, DT, RF, XGBOOST


### Problem Definition  
One of the Portuguese banking institution conducted a marketing campaign based on phone calls from 2008 to 2010. The records of their efforts are available in the form of a dataset. The objective here is to apply machine learning techniques to analyse the dataset and figure out most effective tactics that will help the bank in next campaign to persuade more customers to subscribe to banks term deposit.
The dataset contains various categorical and numerical features with 11162 data sample. The data is labelled. So supervised machine learning algorithms are applicable to this project. The objective is to predict whether the client will subscribe to a term deposit or not. Data pre-processing is done along with suitable exploratory data analysis. Results of various algorithms are compared at the end.

### Methodology  
The dataset class is labelled as ‘yes’ or ‘no’ depending on whether the contacted client has subscribed to the deposit or not. It is a marketing problem and the outcome will largely influence the future strategies of bank. Banking institute has a very large client base and even larger target clients. In real world , less clients will respond positively to marketing campaign and most of them will say no. Contacting all of them is time consuming task and demands tremendous time and efforts. To manage the human resource in efficient way, it is necessary to correctly identify those clients who have more chances of saying yes. This is where machine learning comes into picture.

### Feature Description & Exploratory Data Analysis
The csv file of dataset is obtained from Kaggle website. Originally it has 16 features and one target class. Out of 16 features ,7 are numerical features whereas 9 are categorical features. None of the features contain a null value. Though in this project all the features are considered for classification but still in order to understand the data, some exploratory data analysis have been performed. Observations are as follows. 
1. Age : This is age of client. A violin plot is plotted for age. It shows it is spread as well as histogram. People saying yes has more spread. 
2. Job : This is a categorical feature. It has 12 categories including unknown. The largest clients in this category belong to management jobs with percentage of 50 while unknown category has least count. 
3. Marital : This is a categorical feature. It has 3 categories. Married clients constitute the largest portion of this feature.
4.Education : Maximum clients belong to category of secondary education. The category tertiary education has larger ratio of clients saying yes to term deposit than any other category. 
5. Default : it tells whether the client has credit in bank or not? Most of them don’t have credit. 
6. Balance : It is a numerical feature. It specifies the actual balance in the clients account. 
7. Housing : Whether the client has already got any housing loan from bank? Apparently clients who do not already have housing loan tend to subscribe to deposit more. 
8. Loan : Most of them don’t already have personal loan in bank.
9. Contact : This feature specifies the way of communication. It can be cellular or telephone. Some of them are unknown. 
10. Month : the month in which the client was contacted. Maximum clients were contacted in May. 
11. Day : which day of the month was the client contacted. More clients were contacted in the middle of the month. 
12. Duration : The duration of call in seconds when the client was contacted last time. A histogram is plotted for this feature and there are very less number of clients having very long duration. A violin plot is also given which indicates that people with longer duration have more possibility of saying yes to deposit. 
13. Campaign : Number of times this client was contacted during this campaign. 
14. Pdays : number of days that passed after the client was last contacted in previous campaign. It is value is -1 if the client was not contacted previously. 
15. Previous : Number of times this client was contacted before this campaign.
16. Poutcome : The outcome of previous marketing campaign. It is a categorical feature. Many of them were not contacted in previous campaign. 	
17. Deposit : Whether the clients said yes to subscribe for a term deposit. It has two categories. ‘yes’ or ‘no’.

### Experiment & Discussion
Seven different algorithms are used to solve this problem. Various results have been compared at the end using table. A plot is used to compare ROC curves. Recall is used as one of the performance matrix.
##### Why Recall ?
As It is a marketing problem a lot of resources are included and it is very important to optimise results to save resources. The target variable is ‘deposit’ which reads yes or no based on success or failure of phone calls. Finding out only those clients which have higher chances of saying yes to subscription of term deposit , will save a lot of manhours and efforts. Predicting as many positives as possible out of actual positives from dataset is the goal here, recall has been chosen as one of the performance matrices along with accuracy and AUC score.
##### Hyperparameter Tuning
Selecting the right hyperparameter and it is probable range is a crucial task. Selecting a wider range may cause longer execution time while selecting a narrow range may result in poor tuning of hyperparameters. So enough number of parameters are chosen with enough range to avoid both problems. 
In KNN, K is the hyperparameter which indicates the number of nearest neighbours used by algorithm to do the majority vote and predict result. Naïve Bayes internally uses alpha, which is basically constant of Laplace smoothing. In Logistic Regression, λ is the hyperparameter which controls the amount of regularisation in optimisation. Sklearn implements it as c for uniformity. Here c= 1/ λ. In SVM c is the hyperparameter, which has a very similar purpose but in SVM three kernels are used. Hence RBF kernel has gamma and polynomial kernel has degree as another hyperparameter.
Most important parameter of Decision tree is max_depth. It is higher value may cause overfitting and lower value may cause underfitting. Apart from that min_samples_leaf, min_sample_split and criterion are tuned for Decision tree. In addition to that random forest uses some extra parameters. Bootstrap will decide if samples are bootstraped or not. Max_features decides how many features to choose while doing column sampling. N_estimators will decide how may trees to be included in forest. In this project XGBOOST uses eta, min_child_weight, gamma, subsample, colsample_bytree as hyperparameter. 

### Accuracy given by algorithms
1. KNN	= 0.74
2. Naïve Bayes = 0.70
3. Logistic Regression	=	0.81
4. LR with polynomial Feature degree 2	=	0.84
5. LR with polynomial Feature degree 3	=	0.83
6. SVM	=	0.82
7. Decision Tree	=	0.81
8. Random Forest	=	0.84
9. XGBOOST	=	0.85

### Discussion
For Naïve Bayes there is no need to do the hyperparameter tuning. The algorithm does it internally. For logistic regression it uses L2 regularization to give the best result. In SVM linear kernel gives better results than RBF or Polynomial. It may mean that the data is almost linearly separable. The maximum depth for decision tree is 8. More than that would cause overfitting. Random forest has used 2000 estimators to come to a good result. 
The first two algorithms have lower results. KNN is distance based algorithm. As the number of features increase it is performance decreases due to curse of dimensionality. Naïve Bayes is mostly used for text classification. It is more efficient with categorical features and for numerical features it requires gaussian distribution. Tree based Algorithms are giving better results that others.

### Conclusion
This sums up for the classification task of bank marketing dataset. It is found that XGBOOST gives the best value for accuracy which is 0.85 while SVM gives the best value for recall which is 0.89. The best AUC score of 0.92 comes from Random Forest as well as XGBOOST. The results of K nearest neighbours and Naïve Bayes are less while rest of the algorithms have given more or less same result with minor differences.  
As per the feature importance of XGBOOST it is clear that bank should focus more on clients with success in previous campaign. Whether client uses cellular phone or not and the month in which client is being called play a vital role in prediction. One thing should be noted that  this modelling is based on behaviour of clients and not on their motivations. The features reveal the actions of client but not his/her thought process. So more descriptive features can be useful here for example interview summery. In that case natural language processing will give better results. 
In these times of crisis preserving the relationship with best customers is more crucial than ever. Using these results bank can specifically target clients and gain higher success in their endeavours. Saving a lot of time by not focusing on clients with less probability is yet another advantages of this project.
