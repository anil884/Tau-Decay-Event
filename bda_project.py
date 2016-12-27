#Anil
#Aishwarya 
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import random
import time
##Input Files
train_file ='training.csv'
#test_file = 'test.csv'
check_aggrement = 'check_agreement.csv'
check_corr = 'check_correlation.csv'
#constants
c = 299.792458     # Speed of light 1000 KM/s
m_mu = 105.6583715 # Muon mass (in MeV)
m_tau = 1776.82    # Tau mass (in MeV)
# List of the features for the second classifier (GeoMetric Features):
geo_features = ['FlightDistance','FlightDistanceError','LifeTime','IP','IPSig','VertexChi2','dira','pt','DOCAone','DOCAtwo','DOCAthree','IP_p0p2','IP_p1p2','isolationa',
	'isolationb','isolationc','isolationd','isolatione','isolationf','iso','CDF1','CDF2','CDF3','ISO_SumBDT','p0_IsoBDT','p1_IsoBDT','p2_IsoBDT',
	'p0_track_Chi2Dof','p1_track_Chi2Dof','p2_track_Chi2Dof','p0_IP','p0_IPSig','p1_IP','p1_IPSig','p2_IP','p2_IPSig',
	'E','FlightDistanceSig','DOCA_sum','isolation_sum','IsoBDT_sum','track_Chi2Dof','IP_sum','IPSig_sum','CDF_sum'] #Latent Features -last line
# List of the features for the second classifier (Kinematics Feature):
kn_features = ['dira','pt','p0_pt','p0_p','p0_eta','p1_pt','p1_p','p1_eta','p2_pt','p2_p','p2_eta',
	'E','pz','beta','gamma','beta_gamma','Delta_E','Delta_M','flag_M','E0','E1','E2','E0_ratio','E1_ratio','E2_ratio',
	'p0_pt_ratio','p1_pt_ratio','p2_pt_ratio','eta_01','eta_02','eta_12','t_coll'] #latent feature -last 2 line

#reading a file and storing the data in panda's data frame
def read_csv(file_name):
	'''
	used pandas dataframe to read and store data
	'''
	data = pd.read_csv(file_name, index_col='id')
	return data
#Spilting train and test data into 80/20 ratio
def train_test_split(df):
	'''
		randomly sampling the train data to train and test data set 
	'''
	df0 = df[df['signal']==0]
	df1 = df[df['signal']==1]
	bt0 = int(0.8 * len(df0))
	bt1 = int(0.8 * len(df1))
	train0 = df0.ix[df0.index.values[:bt0]]
	test0 = df0.ix[df0.index.values[bt0:]]
	train1= df1.ix[df1.index.values[:bt1]]
	test1 = df1.ix[df1.index.values[bt1:]]
	train = pd.concat([train0,train1])
	test = pd.concat([test0,test1])
	return train,test
# add latent features:
def add_features(df):
	'''
		Extract Latent features
	'''
	# Kinematic features of the tau particle:
	df['E'] = (df['p0_p']**2 + df['p1_p']**2 + df['p2_p']**2 + 3*(m_mu**2)).apply(np.sqrt)
	df['pz'] = df['p0_pt']*(df['p0_eta']).apply(np.sinh) + df['p1_pt']*(df['p1_eta']).apply(np.sinh) + df['p2_pt']*(df['p2_eta']).apply(np.sinh)
	df['beta_gamma'] = df['FlightDistance'] / (df['LifeTime'] * c )
	df['M_lt'] = (df['pt']**2 + df['pz']**2).apply(np.sqrt) / df['beta_gamma']
	df['M_inv'] = (df['E']**2 - df['pt']**2 - df['pz']**2 )
	df['Delta_E'] = (df['M_lt']**2 + df['pz']**2 + df['pt']**2).apply(np.sqrt) - df['E']
	df['Delta_M'] = df['M_lt'] - df['M_inv']
	df['flag_M'] = np.where((df['M_lt'] - m_tau - 1.44) < 17 , 1, 0)
	df['gamma'] = df['E']/df['M_inv']
	df['beta'] = (1-df['gamma']**2 ).apply(np.sqrt) / df['gamma']
	# Kinematic features of muon particles:
	df['E0'] = (m_mu**2 + df['p0_p']**2).apply(np.sqrt)
	df['E1'] = (m_mu**2 + df['p1_p']**2).apply(np.sqrt)
	df['E2'] = (m_mu**2 + df['p2_p']**2).apply(np.sqrt)
	df['E0_ratio'] = df['p0_p'] / df['E']
	df['E1_ratio'] = df['p1_p'] / df['E']
	df['E2_ratio'] = df['p2_p'] / df['E']
	df['p0_pt_ratio'] = df['p0_eta'] / df['pt']
	df['p1_pt_ratio'] = df['p1_eta'] / df['pt']
	df['p2_pt_ratio'] = df['p2_eta'] / df['pt']
	df['eta_01'] = df['p0_eta'] - df['p1_eta']
	df['eta_02'] = df['p0_eta'] - df['p2_eta']
	df['eta_12'] = df['p1_eta'] - df['p2_eta']
	df['t_coll'] = (df['p0_pt'] + df['p1_pt'] + df['p2_pt'])/df['pt']
	# Geometric features:
	df['FlightDistanceSig'] =  df['FlightDistance']/df['FlightDistanceError']
	df['DOCA_sum'] = df['DOCAone'] + df['DOCAtwo'] + df['DOCAthree']
	df['isolation_sum'] = df['isolationa'] + df['isolationb'] + df['isolationc'] + df['isolationd'] + df['isolatione'] + df['isolationf']
	df['IsoBDT_sum'] = df['p0_IsoBDT'] + df['p1_IsoBDT'] + df['p2_IsoBDT']
	df['track_Chi2Dof'] = (df['p0_track_Chi2Dof']**2 + df['p1_track_Chi2Dof']**2 + df['p2_track_Chi2Dof']**2).apply(np.sqrt)
	df['IP_sum'] =  df['p0_IP'] +  df['p1_IP'] +  df['p2_IP']
	df['IPSig_sum'] = df['p0_IPSig'] + df['p1_IPSig'] + df['p1_IPSig'] 
	df['CDF_sum'] = df['CDF1'] + df['CDF2'] + df['CDF3'] 
	return df

#Compute KS metric
def KS_aggrement_test(agg_prob, data):
	'''
		Simple implementation of 2 sample Kolmogorov–Smirnov test
		p_a0 is real data and p_a1 is mote carlo data as these are mainly simulated data 
	'''
	p_a0 = agg_prob[data['signal'].values == 0]
	p_a1 = agg_prob[data['signal'].values == 1]
	w_a0 = data[data['signal'] == 0]['weight'].values
	w_a1 = data[data['signal'] == 1]['weight'].values
	#normalizing the weights
	w_a0 = w_a0 / sum(w_a0)
	w_a1 = w_a1 / sum(w_a0)
	l = np.concatenate([np.zeros(len(w_a0)),  np.ones(len(w_a1))])
	p = np.concatenate([p_a0,p_a1])
	w = np.concatenate([w_a0,w_a1])
	fpr, tpr, _ = roc_curve(l, p, sample_weight=w)
	ks = np.max(np.abs((fpr - tpr)))
	return ks

def cvm_correlation_test(p,m):
	'''
		Cramer-von Mises metric
		we are using window size as 200 and step size as 50
	'''
	p =  p[np.argsort(m)]
	p =  np.argsort(np.argsort(p))
	k = len(p) -200 +1 #window size 200
	rw = [p[i:i+200]  for i in range(k)][::50] #stepsize 50 
	d = np.array(range(1,len(p)+1)) / len(p) #distribution of the prediction
	cvm= []
	for w in rw:
		sd = np.cumsum(np.bincount(w,minlength=len(p)))
		sd =  sd / sd[-1]
		cvm.append(np.mean((d-sd)**2))
	return np.mean(cvm) 
		
#calculate weighted AUC on actual and predicted data	
def weighted_auc(l,p):
	'''
	l = actual class labels
	p =  predicted labels
	'''
	fpr,tpr,_ = roc_curve(l,p)
	th_steps = np.array([0,0.2,0.4,0.6,0.8,1])
	w = np.array([4,3,2,1,0]) #weights for each area
	w_auc =0
	for  i in range(1,len(w)+1):
		auc_c = auc(fpr,np.minimum(tpr,th_steps[i]),reorder=True)
		auc_p = auc(fpr,np.minimum(tpr,th_steps[i-1]),reorder=True)
		w_auc = w_auc + (auc_c - auc_p)*w[i-1]
	w_auc /= np.sum((th_steps[1:] - th_steps[:-1]) * w)
	return w_auc

#predict output for an input data
def predict(c,d,l,gbc):
	'''
		gives predicted probabality of the input
	'''
	if gbc:
		return c.predict(xgb.DMatrix(d[l]))
	else:
		return c.predict_proba(d[l].values)[:,1]
#Validate and Test classifier for diffeent probability weight
def test_classifier(c1,c2,train,agg,corr,test,name,gbc= False,p=0):
	'''
	Predict for validation and test data set . plot for AUC,Accuracy, KS Metric and cvm
	'''
	list1 = geo_features
	list2 = kn_features
	train_predict1 =  predict(c1,train,list1,gbc)
	train_predict2 =  predict(c2,train,list2,gbc)
	agg_predict1 =  predict(c1,agg,list1,gbc)
	agg_predict2 =  predict(c2,agg,list2,gbc)
	corr_predict1 =  predict(c1,corr,list1,gbc)
	corr_predict2 =  predict(c2,corr,list2,gbc)
	test_predict1 = predict(c1,test,list1,gbc)
	test_predict2 =  predict(c2,test,list2,gbc)
	final_data =[]
	output=[]
	for prob in range(0,101):
		prob=prob/100
		train_predict =  (train_predict1*prob + train_predict2*(1-prob))
		agg_predict = (agg_predict1*prob + agg_predict2*(1-prob))
		corr_predict = (corr_predict1*prob + corr_predict2*(1-prob))
		test_predict = (test_predict1*prob + test_predict2*(1-prob))
		ks =  KS_aggrement_test(agg_predict, agg)
		cvm = cvm_correlation_test(corr_predict, corr['mass'])
		AUC = weighted_auc(test['signal'], test_predict)
		test_final = (test_predict>=0.5).astype(int)
		acc = accuracy_score(test['signal'].values,test_final)
		agg_final = (agg_predict>=0.5).astype(int)
		agg_acc = accuracy_score(agg['signal'].values,agg_final)
		agg_auc = weighted_auc(agg['signal'], agg_predict)
		dd = [prob,ks,cvm,AUC,acc,agg_acc,agg_auc]
		final_data.append(dd)
	final_data=np.array(final_data)
	check1 =  final_data[final_data[:,1] <= 0.09,:]
	check1 = final_data if len(check1) == 0 else check1
	check2 =  check1[check1[:,2] <= 0.002,:]
	min_index = np.argmin(check1[:,2]) if len(check2) == 0 else np.argmax(check2[:,6])
	check2 = check1 if len(check2)==0 else check2
	output = check2[min_index,:]
	test_predict = (test_predict1*output[0] + test_predict2*(1-output[0]))
	test_final = (test_predict>=0.5).astype(int)
	cm = confusion_matrix(test['signal'].values,test_final)
	print('Confusion Matrix for '+name)
	print(cm)
	plot_classifier_validation(final_data,output,name)
	plot_classifier_output(final_data,output,name)
	return final_data,output

#plot validation results
def plot_classifier_validation(final_data,output,name):
	'''
	plot KS score and cvm score for a classifier for different weight probability
	'''
	plt.figure()
	fig = plt.gcf()
	plt.plot(final_data[:,0],final_data[:,1],'r-',label='KS Aggrement Metric')
	plt.plot(final_data[:,0],final_data[:,2],'g-',label='CVM Correlation Metric')
	plt.plot(final_data[:,0],final_data[:,5]/10.0,'b-',label='Test Data Accuracy/10')
	plt.plot(final_data[:,0],final_data[:,6]/10.0,'m-',label='Test Data AUC/10')
	plt.plot(output[0],output[1],'ko')
	plt.plot(output[0],output[2],'ko')
	plt.plot(output[0],output[5]/10.0,'ko')
	plt.plot(output[0],output[6]/10.0,'ko')
	plt.ylabel('Score')
	plt.xlabel('Weight Probability')
	text = 'W='+str(output[0])+', ks='+str(output[1])+', cvm='+str(output[2])+',\n'+'AUC='+str(output[1])+', Accuracy='+str(output[4])
	plt.title(text)
	plt.legend(loc='lower left', shadow=False)
	plt.grid()
	fig_name1 = name + '_TestMetric.png'
	fig.savefig(fig_name1)
	plt.clf()

#plot test results for a classifier
def plot_classifier_output(final_data,output,name):
	'''
	plot Test data Acuuracy score and AUC score for a classifier
	'''
	plt.figure()
	fig1 = plt.gcf()
	plt.plot(final_data[:,0],final_data[:,3],'r-',label='Weighted AUC Score')
	plt.plot(final_data[:,0],final_data[:,4],'g-',label='Accuracy Score')
	plt.plot(output[0],output[3],'ko')
	plt.plot(output[0],output[4],'ko')
	plt.xlabel('Weight Probability')
	plt.ylabel('Score')
	text = name+'\n ( W='+str(output[0])+', AUC='+str(output[3])+', Accuracy='+str(output[4])+')'
	plt.title(text)
	plt.grid()
	plt.legend(loc='lower left', shadow=False)
	fig_name2 = name+'_FinalValidation_output.png'
	fig1.savefig(fig_name2)
	plt.clf()	
		
#plot bar graph for compering between classifier
def plot_bargraph(gb,rf,dt,name,c1,c2,c3):
	'''
	plot comaprision graph for Accuracy and AUC score , KS and CVM score
	'''
	ng=3
	d1 = (gb[1],rf[1],dt[1])
	d2 = (gb[2],rf[2],dt[2])
	plt.figure()
	fig, ax = plt.subplots()
	index = np.arange(ng)
	bar_width = 0.35
	rects1 = plt.bar(index, d1, bar_width,color='r',label='KS Metric')
	rects2 = plt.bar(index+bar_width, d2, bar_width,color='b',label='Correlation Metric')
	plt.xlabel('Classifiers')
	plt.ylabel('Metric Scores')
	plt.title('Testing Metric Scores by classifier')
	plt.xticks(index + bar_width, (c1,c2,c3))
	plt.legend()
	plt.tight_layout()
	fig.savefig(name+'_Testing.png')
	plt.clf()
	#output fig
	plt.figure()
	fig2, ax2 = plt.subplots()
	d3 = (gb[3],rf[3],dt[3])
	d4 = (gb[4],rf[4],dt[4])
	rects3 = plt.bar(index, d3, bar_width,color='r',label='AUC Score')
	rects4 = plt.bar(index+bar_width, d4, bar_width,color='b',label='Accuracy Score')
	plt.xlabel('Classifiers')
	plt.ylabel('Metric Scores')
	plt.title('AUC and Accuracy Scores by classifier')
	plt.xticks(index + bar_width, (c1,c2,c3))
	plt.legend(loc='lower right')
	plt.tight_layout()
	fig2.savefig(name+'.png')	
	plt.clf()

##Start Project work
print('Reading Data from Train,Aggrement_check and correlation files...')
train_data =  read_csv(train_file)
agg_data = read_csv(check_aggrement)
corr_data =  read_csv(check_corr)
print('Creating train and test data set')
train_data,test_data =  train_test_split(train_data)
print('Length of train set:', len(train_data))
print('Length of test set:', len(test_data))
print('Adding latent features to train  data ...')
train_data = add_features(train_data)
print('Adding latent features to Validation  data ...')
test_data = add_features(test_data)
print('Adding latent features to aggrement test data ...')
agg_data = add_features(agg_data)
print('Adding latent features to correlation test data ...')
corr_data =  add_features(corr_data)
#print('Droping NAN elements')
train_data =  train_data.dropna()
agg_data =  agg_data.dropna()
corr_data = corr_data.dropna()
test_data = test_data.dropna()
#gradient boosting algorithm
print('Removing non representative features')
train = xgb.DMatrix(train_data[geo_features],train_data['signal'])
train2 = xgb.DMatrix(train_data[kn_features],train_data['signal'])
xgbParams = {"objective": "binary:logistic",
	"eta": 0.5,
	"max_depth": 4,
	"scale_pos_weight": 3,
	"silent": 1,
	"seed": 1,
	}
print('Training Gradient Boosting')
start = time.time()
gbc = xgb.train(xgbParams, train, 200);
gbc1 = xgb.train(xgbParams, train2, 200);
end = time.time()
print('Training time for Gradient Boosting is: ', end-start, 'secs')
print('Testing and Validating for Gradient Boosting')
gb_output,gb_o = test_classifier(gbc,gbc1,train_data,agg_data,corr_data,test_data,'GradientBoosting',True)
#Random forest
print('Training Random Forest')
start = time.time()
rf = RandomForestClassifier(n_estimators=200)
rf.fit(train_data[geo_features].values,train_data['signal'].values)
rf1 = RandomForestClassifier(n_estimators=100)
rf1.fit(train_data[kn_features].values,train_data['signal'].values)
end = time.time()
print('Training time for Random Forest is: ', end-start, 'secs')
print('Testing and Validating Random Forest')
rf_output,rf_o = test_classifier(rf,rf1,train_data,agg_data,corr_data,test_data,'RandomForest',False)
#Decsion Tree
print('Training on Decision Tree(c4.5) ')
dt = tree.DecisionTreeClassifier(min_samples_split=100)
dt1 = tree.DecisionTreeClassifier(min_samples_split=50)
start = time.time()
dt.fit(train_data[geo_features].values,train_data['signal'].values)
dt1.fit(train_data[kn_features].values,train_data['signal'].values)
end = time.time()
print('Training time for Descision Tree is: ', end-start, 'secs')
print('Testing and Validating Descision tree')
dt_output,dt_o = test_classifier(dt,dt1,train_data,agg_data,corr_data,test_data,'DecsionTree(C4.5)',False)
#Linear Model
print('Training on Linear Model ')
lr = linear_model.LogisticRegression(C=1e5)
lr1 = linear_model.LogisticRegression(C=1e5)
start = time.time()
lr.fit(train_data[geo_features].values,train_data['signal'].values)
lr1.fit(train_data[kn_features].values,train_data['signal'].values)
end = time.time()
print('Training time for Linear Model Logisitic Regression is: ', end-start, 'secs')
print('Testing and Validating Linear Model Logistic regression')
lr_output,lr_o = test_classifier(lr,lr1,train_data,agg_data,corr_data,test_data,'LogisticRegression',False)
#Adaptive Boosting
print('Training on Adaboosting')
bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4),algorithm="SAMME",n_estimators=200)
bdt1 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4),algorithm="SAMME",n_estimators=200)
start = time.time()
bdt.fit(train_data[geo_features].values,train_data['signal'].values)
bdt1.fit(train_data[kn_features].values,train_data['signal'].values)
end = time.time()
print('Training time for AdaBoosting is: ', end-start, 'secs')
print('Testing and Validating AdaBoosting')
bdt_output,bdt_o = test_classifier(bdt,bdt1,train_data,agg_data,corr_data,test_data,'AdaBoosting_dt',False)
#Naive Bayes
print('Training on Naive Bayes')
nb = GaussianNB()
nb1 = GaussianNB()
start = time.time()
nb.fit(train_data[geo_features].values,train_data['signal'].values)
nb1.fit(train_data[kn_features].values,train_data['signal'].values)
end = time.time()
print('Trainig time for Naive Bayes is: ', end-start, 'secs')
print('Testing and Validating Naive Bayes')
nb_output,nb_o = test_classifier(nb,nb1,train_data,agg_data,corr_data,test_data,'NaiveBayes',False)
##Showing all results in bar graph
plot_bargraph(gb_o,rf_o,bdt_o,'Ensemble','GradientBoosting','RandomForest','AdaBoosting')
plot_bargraph(lr_o,nb_o,dt_o,'NonEnsemble','LogisticRegresion','NaiveBayes','DecisionTree(C4.5)')
print('Gradient Boosting(Weighted AUC,Accuracy):',gb_o[[3,4]])
print('Random Forest(Weighted AUC,Accuracy):',rf_o[[3,4]])
print('AdaBoosting using Descision Tree (Weighted AUC,Accuracy):',bdt_o[[3,4]])
print('Descision Tree(Weighted AUC,Accuracy):',dt_o[[3,4]])
print('Naive Bayes(Weighted AUC,Accuracy):',nb_o[[3,4]])
print('Logistic Regression(Weighted AUC,Accuracy):',rf_o[[3,4]])
train_data.to_csv('FinalTraining.csv', sep=',')
#import pdb;pdb.set_trace();