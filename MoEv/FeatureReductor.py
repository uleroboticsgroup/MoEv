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
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class FeatureReductor:
	def __init__(self):
		pass


	def calculatePCAComponents(self, X):
		pca = PCA()
		pca.fit(X)
		cumsum = np.cumsum(pca.explained_variance_ratio_)
		print(cumsum)
		d = np.argmax(cumsum >= 0.95) + 1
		plt.figure(figsize=(6,4))
		plt.plot(cumsum, linewidth=3)
		plt.axis([0, 10, 0, 1])
		plt.xlabel("Dimensions")
		plt.ylabel("Explained Variance")
		plt.plot([d, d], [0, 0.95], "k:")
		plt.plot([0, d], [0.95, 0.95], "k:")
		plt.plot(d, 0.95, "ko")
		plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
             arrowprops=dict(arrowstyle="->"), fontsize=16)
		plt.grid(True)
		#save_fig("explained_variance_plot")
		plt.show()
		return d

	def reductionPCA_one(self, X, k, random_state_value ,whiten_value):
		pca = PCA(n_components=k, random_state=random_state_value, whiten=whiten_value)
		Y = pca.fit_transform(X)
		#print(Y)
		return Y

	def reductionPCA_auto(self, X, X_test, k, random_state_value ,whiten_value):
		pca = PCA(n_components=k, random_state=random_state_value, whiten=whiten_value)
		X = pca.fit_transform(X)
		X_test = pca.transform(X_test)
		#print(Y)
		return X, X_test
		
	#Return a transformed X
	def selectKBest(self, function, value, X, y):
		if function == "f_regression":
			return pd.DataFrame(SelectKBest(f_regression, k=value).fit_transform(X, y))
		if function == "chi2":
			print(SelectKBest(chi2, k=value).fit_transform(X, y))

	def extraTreeClassifier(self, X, y):
		clf = ExtraTreesClassifier(n_estimators=50)
		clf = clf.fit(X, y)
		model = SelectFromModel(clf, prefit=True)
		newX = model.transform(X)
		print (newX)
		return newX

