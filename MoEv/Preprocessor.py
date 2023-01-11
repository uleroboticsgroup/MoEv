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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

class Preprocessor:
	
	def __init__(self):
		pass


	# We remove from the DataFrame the first row and the label column so that they are not treated
	def adapt_dataset(self, df):
		#We save the data in the label column
		labelColumn = df['Label']
		#Remove the column from the dataframe
		f_WithOut_Label = df.drop('Label', axis=1)
		#We save the name of all columns by removing the label field
		columns_names = df.keys().drop('Label')
		return columns_names,labelColumn,f_WithOut_Label


	def variance(self, df, limit, list):
		for col in df:
			flag = False
			for i in list:
				if col == i:
					flag = True
			if not flag:
				variance = df[col].var()
				if variance == limit:
					print("Columna: %s con varianza %f" % (col, variance))
					df = df.drop(col, axis=1)
		return df

	#Apply normalization method
	def minScalerNormalization(self, X):
		columnas = X.columns
		scaler = MinMaxScaler()
		scaler.fit(X)
		X_scaled = scaler.transform(X)
		X_scaled = pd.DataFrame(X_scaled, columns = columnas)
		return X_scaled

	#Apply normalization method
	def minScalerNormalizationAuto(self, X, X_test):
		scaler = MinMaxScaler()
		X_scaled = scaler.fit_transform(X)
		X_test_scaled = scaler.transform(X_test)
		return X_scaled, X_test_scaled
		#Apply normalization method
	
	def maxScalerNormalization(self, X):
		scaler = MaxAbsScaler()
		scaler.fit(X)
		X_scaled = scaler.transform(X)
		return X_scaled

	def maxScalerNormalizationAuto(self, X, X_test):
		scaler = MaxAbsScaler()
		X_scaled = scaler.fit_transform(X)
		X_test_scaled = scaler.transform(X_test)
		return X_scaled, X_test_scaled
	

	#Apply standardization method
	def standardScaler(self, X):
		scaler = StandardScaler()
		newX = scaler.fit_transform(X)
		return newX

	def robustScaler(self, X):
		columnas = X.columns
		scaler = RobustScaler()
		X_scaled = scaler.fit_transform(X)
		X_scaled = pd.DataFrame(X_scaled, columns = columnas)
		return X_scaled

	def robustScalerAuto(self, X, X_test):
		scaler = RobustScaler()
		X_scaled = scaler.fit_transform(X)
		X_test_scaled = scaler.transform(X_test)
		return X_scaled, X_test_scaled
	
	#Apply standardization method
	def standardScalerAuto(self, X, X_test):
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)
		X_test_scaled = scaler.transform(X_test)
		return X_scaled, X_test_scaled
