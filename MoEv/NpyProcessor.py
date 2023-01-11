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
import matplotlib.pyplot as plt
import pprint
import RGB2GrayTransformer
import HogTransformer


class NpyProcessor:


	def __init__(self):
		pass

	def npy_processor(self, conf):
		pp = pprint.PrettyPrinter(indent=4)

		

		data_raw = np.load(conf["npy"]["data_raw"])
		data_label = np.load(conf["npy"]["data_label"])

		print(data_label.shape)
		print(data_raw.shape)

		from sklearn.model_selection import train_test_split
		 
		X_train, X_test, y_train, y_test = train_test_split(
		    data_raw,
		    data_label,
		    test_size=0.2,
		    shuffle=True,
		    random_state=42,
		)
		print(X_train.shape)
		print(y_train.shape)
		print(X_test.shape)
		print(y_test.shape)


		from sklearn.linear_model import SGDClassifier
		from sklearn.model_selection import cross_val_predict
		from sklearn.preprocessing import StandardScaler
		import skimage
		 
		# create an instance of each transformer
		grayify = RGB2GrayTransformer.RGB2GrayTransformer()
		hogify = HogTransformer.HogTransformer(
		    pixels_per_cell=(8, 8),
		    cells_per_block=(2,2),
		    orientations=9,
		    block_norm='L2-Hys'
		)
		scalify = StandardScaler()
		 
		# call fit_transform on each transform converting X_train step by step
		X_train_gray = grayify.fit_transform(X_train)
		X_train_hog = hogify.fit_transform(X_train_gray)
		X_train_prepared = scalify.fit_transform(X_train_hog)
		
		X_test_gray = grayify.transform(X_test)
		X_test_hog = hogify.transform(X_test_gray)
		X_test_prepared = scalify.transform(X_test_hog)

		return X_train_prepared, X_test_prepared, y_train, y_test

		print(X_train_prepared.shape)

