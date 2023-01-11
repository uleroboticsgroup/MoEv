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
import glob
import pandas as pd
import numpy as np
import argparse
import datetime
from sklearn import preprocessing

#Function to convert a date to its posix format
def to_Posix(date):
	date = pd.to_datetime(date.strip(), infer_datetime_format=True)
	return date.timestamp()

#Function to convert a series of IPs to integer values
def map_ip(new_dataset):
	for i in new_dataset['Src IP']:
		if not type(i) == int:
			o = map(int, i.split('.'))
			res = ((o[0] * pow(256, 3)) + (o[1] * pow(256, 2)) + (o[2] * 256) + o[3])
			new_dataset["Src IP"].replace(i, value=res, inplace=True)
	for i in new_dataset['Dst IP']:
		if not type(i) == int:
			o = map(int, i.split('.'))
			res = ((o[0] * pow(256, 3)) + (o[1] * pow(256, 2)) + (o[2] * 256) + o[3])
			new_dataset["Dst IP"].replace(i, value=res, inplace=True)

	return new_dataset

def analysis(dataset):
	rows_num = len(dataset)
	for col in dataset:
		#We calculate if the columns have values ​​that do not vary
		aux = dataset[col][0] * rows_num
		col_sum = dataset[col].astype(float).sum()
		if not aux == 0:
			if float(col_sum)/float(aux) == 1:
				print("All values of column %s are equals to: %f" % (col, float(dataset[col][0])))
				if args.mode == 2:
					dataset = dataset.drop(columns=[col], inplace=True)
		if col_sum == 0:
				print("All values of column %s are equals to: %f" % (col, float(dataset[col][0])))
				if args.mode == 2:
					dataset = dataset.drop(columns=[col], inplace=True)
		#Pair of negatives if necessary
		negative_list = dataset[col] < 0
		negatives_num = negative_list.sum()
		if not negatives_num == 0:
			print("Column %s has %i negative values" % (col, negatives_num))
			#Pair of negatives if necessary
		#We calculate the NaN values
		nan_list = dataset[col].isnull()
		nan_num = nan_list.sum()
		if not nan_num == 0:
			print("Column %s has %i NaN values" % (col, nan_num))
			dataset = dataset.fillna(0)
	
	return dataset




#Parser to interpret command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Source csv path")
parser.add_argument("-m", "--mode", help="Mode: analysis (1), preprocessor (0)")
parser.add_argument("-t", "--type", help="Rescaling type: normalization (norm), standardization (std) or no rescaling (noscl)")
parser.add_argument("-l", "--labeled", help="Add column label 1 | do not add it 0")
parser.add_argument("-o", "--output", help="Destination csv file path")
args = parser.parse_args()

if args.file:
	dataset = pd.read_csv(args.file)
else:
	print("Error. A file in csv format is required")
	parser.print_help()
	quit()

# We filter the rows that have moved after applying CICFlowMeter
num_fields_list = dataset.count(axis='columns')

index_list = num_fields_list[num_fields_list < 82].index.values.astype(int) #If you have less than 82 fields then you have an error
for i in index_list:
	dataset = dataset.drop([i],axis=0) #We delete the row that contains the error

#We delete the Flow ID column
dataset = dataset.drop(columns=['Flow ID'])

timestamp = dataset['Timestamp']
for j in timestamp:
	if not type(j) == float:
		dataset['Timestamp'].replace(j, value=to_Posix(j), inplace=True)

dataset = map_ip(dataset)


if args.mode:
	dataset = analysis(dataset)

if not args.type:
	quit()

if args.type == "norm":
	final_dataset = preprocessing.normalize(dataset)
if args.type == "std":
	scaler = preprocessing.StandardScaler().fit(dataset)
	final_dataset = scaler.transform(dataset)
if args.type == "noscl":
	dataset.to_csv("./noscl.csv", index=False, encoding='utf-8-sig')
	quit()

np.savetxt(args.output, final_dataset, delimiter=',')

