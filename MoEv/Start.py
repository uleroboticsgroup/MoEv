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
import MoEv
import argparse
import Preprocessor
import FeatureReductor
import NpyProcessor

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="Type of data: cic, netflow or npy ")
args = parser.parse_args()
data_type = ""

#Check the data that is going to loaded
if not args.type:
	print("It is necessary to indicate the type of data")
	parser.print_help()
	quit()
else:
	#CICFlowMeter data
	if args.type == "cic":
		data_type = args.type
	#Netflow data
	elif args.type == "netflow":
		data_type = args.type
	#npy images data
	elif args.type == "npy":
		data_type = args.type
	else:
		print("Incorrect type of data")
		parser.print_help()
		quit()



#We create Moev class that contain all methods to clean and analize the data and create and testing the models 
npy_test = NpyProcessor.NpyProcessor()
moev_Manager = MoEv.MoEv("conf.yaml", 0.33, None, "Label", data_type)
conf_file = moev_Manager.get_conf_file()
#Check if is posible launch MoEv
if conf_file["autotest"]["enabled"] and conf_file["testSaveModels"]["enabled"]:
	print("Incompatible options  please disable Autotest or testSaveModels")
elif conf_file["Models"]["Save_Models"]["enabled"] and conf_file["testSaveModels"]["enabled"]:
	print("Incompatible options  please disable testSaveModels or Save_Models")
else:
	#If data is images we should process the data befere create and test the models
	if args.type == "npy":
		#Process data
		X_train, X_test, y_train, y_test = npy_test.npy_processor(conf_file)
		#Load data externally
		moev_Manager.load_data(X_train, X_test, y_train, y_test)
		#Create and test models
		moev_Manager.createModels()
	else:
		moev_Manager.load_dataset(conf_file)
		moev_Manager.cleanDataset()
		moev_Manager.analyzeDataset()
		moev_Manager.preprocessDataset()
		moev_Manager.reductFeatures()
		moev_Manager.createModels()
		pass
