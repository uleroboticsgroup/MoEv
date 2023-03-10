#
#
# Copyright (c) 2020 Adrian Campazas Vega, Ignacio Samuel Crespo Martinez, Angel Manuel Guerrero Higueras.
#
# This file is part of MoEv 
#
# MoEv is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MoEv is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from matplotlib.pyplot import figure
from sklearn import metrics

def fit(classifier, X_train, y_train):
	classifier.fit(X_train, y_train)

def predict(classifier, X_test):
	return classifier.predict(X_test)
	

def accuracyScore(y_test, predictions):
	return accuracy_score(y_test, predictions)

def learning_curves(estimator,X, y):
	title = "Learning Curves "
	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
	train_sizes=np.linspace(.1, 1.0, 5)
	ylim=(0.7, 1.01)
	n_jobs=1
	plt.figure()
	plt.title(title)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
		estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					 train_scores_mean + train_scores_std, alpha=0.1,
					 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
			 label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
			 label="Cross-validation score")

	plt.legend(loc="best")

	plt.show()

def classificationReport(predictions, Y_test):
	print("Classification report:\n\n%s \n" % classification_report(Y_test, predictions, digits=6))

def confusionMatrix (predictions, Y_test, classes, normalize=False, title='Confusion matrix'):
	
	
	# calculate confusion matrix

	print("Confusion Matrix:\n")
	print(confusion_matrix(Y_test, predictions))
def cohenKappaScore(predictions, Y_test):
	print("Cohen Kappa Score:\n\n%s \n" % cohen_kappa_score(Y_test, predictions))

def matthewsCorrcoef(predictions, Y_test):
	print("Matthews Corrcoef\n\n%s \n" % matthews_corrcoef(Y_test, predictions))	


def plotConfusionMatrix(model_name,cm, display_labels):
	# calculate confusion matrix
	font = {'family' : 'normal',
			'weight' : 'normal',
			'size'   : 22}


	np.set_printoptions(precision=14)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm,
	                             display_labels=display_labels)
	#normalize=normalize,
	plt.rc('font', **font)
	disp.plot(cmap=plt.cm.Greens)
	fig = plt.gcf()
	fig.set_size_inches(7, 5.5)
	fig.savefig("../figuras/MC_"+model_name + '.jpg')
	#plt.show()


def plotRocCurve(model,y_train,y_test):
	#Dataset con las features y los scores que no son 0,1
	#TODO Probar que esto funciona con modelos supervisados o ver si es mejor usar predict_proba
	if model.modelName == "One_Class_SVM" or model.modelName == "One_Class_SVM.joblib"  or model.modelName == "Isolation_Forest" or model.modelName == "Isolation_Forest.joblib" or model.modelName == "Local_Outlier_Factor" or model.modelName == "Local_Outlier_Factor.joblib":
		scoring = - model.decision_function(y_train)
		fp, tp, thresholds = metrics.roc_curve(y_test, scoring)
	elif model.modelName == "One_Vs_Rest_Classifier" or model.modelName == "One_Vs_Rest_Classifier.joblib" or model.modelName == "SGD_Classifier.joblib" or model.modelName == "SGD_Classifier":
		scoring =  model.decision_function(y_train)
		fp, tp, thresholds = metrics.roc_curve(y_test, scoring)
	else:
		scoring = model.predict_proba(y_train)
		fp, tp, thresholds = metrics.roc_curve(y_test, scoring[:, 1])
	roc_auc = metrics.auc(fp, tp)
	display = metrics.RocCurveDisplay(fpr=fp, tpr=tp, roc_auc=roc_auc)
	display.plot()
	fig = plt.gcf()
	fig.set_size_inches(9.5, 7.5)
	fig.savefig("../figuras/ROC_"+model.modelName + '.jpg', dpi=100)
	#plt.show()


def novelty_Score(estimator,X,y):
	predictions = estimator.predict(X)
	for i in range(len(predictions)):
			if predictions[i] == 1:
				#print("normal")
				predictions[i] = 0
			if predictions[i] == -1:
				#print("anomalia")
				predictions[i] = 1
	print(accuracyScore(y, predictions))
	print(confusion_matrix(y,predictions))
	return accuracyScore(y, predictions)


def gridSearch(classifier, X, y, parameters,model_type):
	#parameters: list of parameters in the configuration file
	if model_type == "novelty":
		clf = GridSearchCV(classifier, parameters, cv=5, scoring=novelty_Score)
	else:
		clf = GridSearchCV(classifier, parameters, cv=5)
	#.values only take the values and not the features name	if we use PCA the values does not 
	clf.fit(X, y)
	print(clf.best_params_)
	return clf




