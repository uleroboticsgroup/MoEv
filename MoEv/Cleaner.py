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
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, filename='MoEv.log', format='%(asctime)s: %(levelname)s - %(message)s')
logger = logging.getLogger('MoEv')
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s')
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('MoEv').addHandler(console)
logger = logging.getLogger(__name__)

class Cleaner:
	def __init__(self):
		pass

	def date_to_posix(self, value):
		value = pd.to_datetime(value.strip(), infer_datetime_format=True)
		return value.timestamp()

	def ip_to_integer(self, ip):
		stripped_ip = list(map(int, ip.split('.')))

		return ((stripped_ip[0] * pow(256, 3)) + (stripped_ip[1] * pow(256, 2)) + (stripped_ip[2] * 256) + stripped_ip[3])

	def parse_time(self, df, target):
		logging.info("Process start: parse date into integer (POSIX)")
		timestamp_list = df['Timestamp']
		for i in timestamp_list:
			if not type(i) == float:
				df['Timestamp'].replace(i, value=self.date_to_posix(i), inplace=True)
		logging.info("Process finished: parse date into integer (POSIX)")
		return df

	def map_ips(self, df, target):
		logging.info("Process start: parse IP into integer")
		for i in df[target]:
			if not type(i) == int:
				df[target].replace(i, value=self.ip_to_integer(i), inplace=True)
		logging.info("Process finished: parse IP into integer")
		return df

	def fix_CIC_error(self, df):
		logging.info("Process start: Fix the CICFlowMeter error")
		field_list = df.count(axis='columns')
		index_list = field_list[field_list < 82].index.values.astype(int)
		for i in index_list:
			df.drop([i], axis=0)
		logging.info("Process finished: Fix the CICFlowMeter error")
		return df

	def search_NaN(self, df):
		logging.info("Searching for NaN values...")
		row_num = len(df)
		for col in df:
			nan_list = df[col].isnull()
			nan_num = nan_list.sum()
			if not nan_num == 0:
				logging.info("Column %s have %i NaN values" % (col, nan_num))
		logging.info("Search completed")
		return df.fillna(0)

	def search_infinite(self, df, lista):
		logging.info("Searching for infinite values...")
		row_num = len(df)
		for col in df:
			flag = False
			for i in lista:
				if(col == i):
					flag = True
			if not flag:
				inf_list = np.isinf(df[col].astype('float64'))
				inf_num = inf_list.sum()	
				if not inf_num == 0:
					aux = df[col].astype('float64').replace([np.inf, -np.inf], np.nan)
					aux = aux.dropna()
					mean = aux.mean()
					df[col] = df[col].astype('float64').replace([np.inf, -np.inf], mean)

					logging.info("Column %s have %i infinite values" % (col, inf_num))
					
		logging.info("Search completed")
		return df

	def search_negatives(self, df):
		logging.info("Searching for negatives values...")
		row_num = len(df)
		for col in df:
			neg_list = df[col] < 0
			neg_num = neg_list.sum()
			if not neg_num == 0:
				logging.info("Column %s have %i negative values" % (col, neg_num))
		logging.info("Search completed")
		return df

	def search_unimportant(self, df):
		logging.info("Searching for columns with no variance...")
		row_num = len(df)
		for col in df:
			aux = df[col][0] * row_num
			col_sum = df[col].astype(float).sum()
			if not aux == 0:
				if float(col_sum)/float(aux) == 1:
					logging.info("All values in column %s are the same: %f" % (col, float(df[col][0])))
				#Tratar las columnas repetidas
			if col_sum == 0:
				print("All values in column %s are 0" % (col)) 
				#Tratar columnas repetidas y de valor 0
		logging.info("Search completed")
		return df

	def duplicatedRows(self, df):
		logging.info("Removing duplicated Rows")
		duplicatedList = df[df.duplicated()]
		index_list = duplicatedList.index.values.astype(int)
		for i in index_list:
			df = df.drop([i], axis=0)
			logging.info("Row %i removed" % (i))

		return df

	def clean_bad_columns(self, df):
		logging.info("Process start: remove bad columns")
		return df.drop(columns=['Flow ID','Flow Byts/s', 'Flow Pkts/s'])

	def removeFeatures(self, df, list):
		logging.info("Removing features...")
		df = df.drop(columns=list)
		return df
		
