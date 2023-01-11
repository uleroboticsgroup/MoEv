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
import os
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import argparse
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
#kBEST
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Source csv path")
parser.add_argument("-m", "--mode", help="Program mode: 0 -> Null detection | 1 -> Analysis of characteristics")
parser.add_argument("-t", "--type", help="Type of extraction: corr -> variable correlation |")
args = parser.parse_args()

dataset = pd.read_csv(args.file)

if not args.mode:
	print("Error. It is necessary to indicate the program mode")
	parser.print_help()
	quit()

if not args.type:
	print("Error. It is necessary to indicate the type of analysis")
	parser.print_help()
	quit()

if args.mode == "0":
	for i in dataset:
		column = dataset[i]
		res = column.isnull().any().any()
		print("Column: ", i, " Value: ", res) #If the result is true then there are null values
		if res:
			qqplot(column.astype(float))
		
	pyplot.show()
	quit()

if args.mode != "1":
	print("Error. Unrecognized mode")
	parser.print_help()
	quit()

#Feature Matrix
X = dataset.drop(columns=['Label'])   
	#Target Variable
Y = dataset["Label"]   

#Extraction of characteristics correlation matrix with the traffic variable
if args.type == "corr":
	       

	cor = dataset.corr()
	cor_target = abs(cor["Label"])
	#Selecting highly correlated features
	relevant_features_05 = cor_target[cor_target>0.5]
	relevant_features_03 = cor_target[cor_target>0.3]

	print("Features with correlation greater than 0.5 -->", relevant_features_05)
	print("Features with correlation greater than 0.3 -->", relevant_features_03)

	newdat = dataset[["Src Port", "Dst Port", "Flow Duration", "Fwd Pkt Len Mean", "Bwd Pkt Len Max", "Bwd Pkt Len Mean", "Flow IAT Std", "Flow IAT Max", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd PSH Flags", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "PSH Flag Cnt", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Idle Mean", "Idle Max", "Idle Min"]]
	corr_Aux = newdat.corr()
	print("Src Port")
	print(abs(corr_Aux["Src Port"]))
	print("Dst Port")
	print(abs(corr_Aux["Dst Port"]))
	print(abs(corr_Aux["Flow Duration"]))
	print(abs(corr_Aux["Fwd Pkt Len Mean"]))
	print(abs(corr_Aux["Bwd Pkt Len Max"]))
	print(abs(corr_Aux["Bwd Pkt Len Mean"]))
	print(abs(corr_Aux["Flow IAT Std"]))
	print(abs(corr_Aux["Flow IAT Max"]))
	print(abs(corr_Aux["Bwd IAT Tot"]))
	print(abs(corr_Aux["Bwd IAT Mean"]))
	print(abs(corr_Aux["Bwd IAT Std"]))
	print(abs(corr_Aux["Bwd IAT Max"]))
	print(abs(corr_Aux["Bwd PSH Flags"]))
	print(abs(corr_Aux["Pkt Len Max"]))
	print(abs(corr_Aux["Pkt Len Mean"]))
	print(abs(corr_Aux["Pkt Len Std"]))
	print(abs(corr_Aux["PSH Flag Cnt"]))
	print(abs(corr_Aux["Pkt Size Avg"]))
	print(abs(corr_Aux["Fwd Seg Size Avg"]))
	print(abs(corr_Aux["Bwd Seg Size Avg"]))
	print(abs(corr_Aux["Idle Mean"]))
	print(abs(corr_Aux["Idle Max"]))
	print(abs(corr_Aux["Idle Min"]))

if args.type == "bacw":

	cols = list(X.columns)
	pmax = 1
	while (len(cols)>0):
		p= []
		dataset_col = X[cols]
		dataset_col_1 = sm.add_constant(dataset_col, has_constant='add')
		print(dataset_col_1)
		model = sm.OLS(Y, dataset_col_1).fit()
		p = pd.Series(model.pvalues.values[1:], index = cols)      
		pmax = max(p)
		feature_with_p_max = p.idxmax()
		if(pmax>0.05):
			cols.remove(feature_with_p_max)
		else:
			breakselected_features_BE = cols

	print(selected_features_BE)

if args.type == "RFE":
	print("RFE")

if args.type == "kbst":
	X = X.drop(columns=["Flow Byts/s"])
	X_new = SelectKBest(chi2, k=2).fit_transform(X, Y)
	print(X_new)
	print(X_new.shape())
quit()
