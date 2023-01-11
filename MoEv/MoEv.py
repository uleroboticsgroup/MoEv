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

logger.info('Moev.py started')
import yaml
import importlib
import os
import pandas as pd
import Cleaner
import Preprocessor 
import FeatureReductor
from sklearn import model_selection
from joblib import dump, load
import TestModels as test
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score




class MoEv:
	
	def __init__(self, file, size, seed, target, data_type):
		with open(file) as f:
			self.conf_file = yaml.load(f, Loader=yaml.FullLoader)
		logging.info("File charged: conf.yaml")
		self.data_type = data_type
		self.models = []
		self.size = size
		self.seed = seed
		self.target = target
		
		
	#We load the dataset if the type is cic or netflow, we save X and Y
	def load_dataset(self,conf):
		conf = self.get_conf_file()
		#Si esta el autotest cargamos ambos df para entrenar con uno y validar con el otro
		if conf["autotest"]["enabled"] == True:
			self.train_path = conf["autotest"]["train_path"]
			self.test_path = conf["autotest"]["test_path"]
			self.output_path = conf["flows"]["output_path"]
			self.set_df (pd.read_csv(self.train_path))
			self.set_dftest (pd.read_csv(self.test_path))
			self.set_X(self.df.drop(columns=[self.target]))
			self.set_y(self.df[self.target])
			self.set_X_dftest(self.get_dftest().drop(columns=[self.target]))
			self.set_y_dftest(self.get_dftest()[self.target])
		else:
			self.input_path = conf["flows"]["input_path"]
			self.output_path = conf["flows"]["output_path"]
			self.df = pd.read_csv(self.input_path)
			self.set_X(self.df.drop(columns=[self.target]))
			self.set_y(self.df[self.target])
			logging.info("Dataset loaded")
		if conf["cleanData"]["enabled"] or conf["analyzeData"]["enabled"]:
			self.cleaner = Cleaner.Cleaner()
			logging.info("Cleaner object created. Make sure that function cleanDataset() is called in start script.")
		else:
			self.cleaner = None


	def get_importObject(self, kls):
		parts = kls.split('.')
		module = ".".join(parts[:-1])
		m = __import__( module )
		for comp in parts[1:]:
			m = getattr(m, comp)
		return m

	def to_csv(self):
		conf = self.get_conf_file()
		#Juntamos las label con los datos para exportar el dataset
		if conf["featureReduction"]["enabled"] or conf["preprocess"]["enabled"]:
			df = pd.DataFrame(self.get_X())
			df.insert(df.shape[1],"Label",self.get_y())
			if conf["autotest"]["enabled"] == True:
				dftest = pd.DataFrame(self.get_X_dftest())
				dftest.insert(dftest.shape[1],"Label",self.get_y_dftest())
				df.to_csv(self.output_path+"_train.csv", index=False, encoding='utf-8-sig')
				dftest.to_csv(self.output_path+"_test.csv", index=False, encoding='utf-8-sig')
			else:
				df.to_csv(self.output_path+".csv", index=False, encoding='utf-8-sig')
		else:
			if conf["autotest"]["enabled"] == True:
				self.get_df().to_csv(self.output_path+"_train.csv", index=False, encoding='utf-8-sig')
				self.get_dftest().to_csv(self.output_path+"_test.csv", index=False, encoding='utf-8-sig')
			else:
				self.get_df().to_csv(self.output_path+".csv", index=False, encoding='utf-8-sig')

		#Esta funcion solo se llama despues de  clean, ya que utiliza todo el df y hay que actualizar X e Y
	def update_X_y(self):
		conf = self.get_conf_file()
		if conf["autotest"]["enabled"] == True:
			self.set_X(self.df.drop(columns=[self.target]))
			self.set_y(self.df[self.target])
			self.set_X_dftest(self.get_dftest().drop(columns=[self.target]))
			self.set_y_dftest(self.get_dftest()[self.target])
			logging.info("Atributtes X,X_dftest, y and y_dftest of class MoEv has changed")
		else:
			self.set_X(self.df.drop(columns=[self.target]))
			self.set_y(self.df[self.target])
			logging.info("Atributtes X and y of class MoEv has changed")

	def splitDf(self, size, seed):
		#self.update_X_y(self.target)
		conf = self.get_conf_file()
		if conf["autotest"]["enabled"]:
			X_train, X_test, y_train, y_test = train_test_split(self.get_X(), self.get_y(), test_size=size, random_state=seed)
			self.set_X_train(self.get_X())
			#self.set_X_test(X_test)
			self.set_y_train(self.get_y())
			#self.set_y_test(y_test)
			self.set_X_dftest(self.get_X_dftest()) 
			self.set_y_dftest (self.get_y_dftest())
			logging.info("Dataframe splitted")
		else:
			X_train, X_test, y_train, y_test = train_test_split(self.get_X(), self.get_y(), test_size=size, random_state=seed)
			self.set_X_train(X_train)
			self.set_X_test(X_test)
			self.set_y_train(y_train)
			self.set_y_test(y_test)
			logging.info("Dataframe splitted")

	def cleanDataset(self):
		conf = self.get_conf_file()
		df = self.get_df()
		if not conf["cleanData"]["enabled"] or self.cleaner == None:
			logging.error("Clean option not activated. Change configuration file.")
		else:
			if self.data_type == "cic":
				if conf["autotest"]["enabled"] == True:
					dftest = self.get_dftest()
					if conf["cleanData"]["fixCICError"]:
						df = self.cleaner.fix_CIC_error(df)
						dftest = self.cleaner.fix_CIC_error(dftest)
						logging.info("CICFlow Error removed")
					if conf["cleanData"]["removeBadColumns"]:
						df = self.cleaner.clean_bad_columns(df)
						dftest = self.cleaner.clean_bad_columns(dftest)
						logging.info("Bad columns removed")
					if conf["cleanData"]["parseIP"]:
						df = self.cleaner.map_ips(df, "Src IP")
						dftest = self.cleaner.map_ips(dftest, "Src IP")
						logging.info("Source IPs parsed to integer")
						df = self.cleaner.map_ips(df, "Dst IP")
						dftest = self.cleaner.map_ips(dftest, "Dst IP")
						logging.info("Destination IPs parsed to integer")
					if conf["cleanData"]["parseDate"]:
						df = self.cleaner.parse_time(df, "Timestamp")
						dftest = self.cleaner.parse_time(dftest, "Timestamp")
						logging.info("Timestamp parsed to POSIX format")
					if conf["cleanData"]["removeFeature"]["enabled"]:
						df = self.cleaner.removeFeatures(df, conf["cleanData"]["removeFeature"]["list"])
						dftest = self.cleaner.removeFeatures(dftest, conf["cleanData"]["removeFeature"]["list"])
						logging.info("Features removed")
					if conf["cleanData"]["removeDuplicatedRows"]:
						df = self.cleaner.duplicatedRows(df)
						dftest = self.cleaner.duplicatedRows(dftest)
						logging.info("Duplicated rows removed")
					self.set_df(df)
					self.set_dftest(dftest)
				else:
					if conf["cleanData"]["fixCICError"]:
						df = self.cleaner.fix_CIC_error(df)
						logging.info("CICFlow Error removed")
					if conf["cleanData"]["removeBadColumns"]:
						df = self.cleaner.clean_bad_columns(df)
						logging.info("Bad columns removed")
					if conf["cleanData"]["parseIP"]:
						df = self.cleaner.map_ips(df, "Src IP")
						logging.info("Source IPs parsed to integer")
						df = self.cleaner.map_ips(df, "Dst IP")
						logging.info("Destination IPs parsed to integer")
					if conf["cleanData"]["parseDate"]:
						df = self.cleaner.parse_time(df, "Timestamp")
						logging.info("Timestamp parsed to POSIX format")
					if conf["cleanData"]["removeFeature"]["enabled"]:
						df = self.cleaner.removeFeatures(df, conf["cleanData"]["removeFeature"]["list"])
						logging.info("Features removed")
					if conf["cleanData"]["removeDuplicatedRows"]:
						df = self.cleaner.duplicatedRows(df)
						logging.info("Duplicated rows removed")
					self.set_df(df)
			else:
				if conf["autotest"]["enabled"] == True:
					dftest = self.get_dftest()
					if conf["cleanData"]["parseIP"]:
						df = self.cleaner.map_ips(df, "srcaddr")
						dftest = self.cleaner.map_ips(dftest, "srcaddr")
						logging.info("Source IPs parsed to integer")
						df = self.cleaner.map_ips(df, "dstaddr")
						dftest = self.cleaner.map_ips(dftest, "dstaddr")
						logging.info("Destination IPs parsed to integer")
						df = self.cleaner.map_ips(df, "nexthop")
						dftest = self.cleaner.map_ips(dftest, "nexthop")
						logging.info("NextHop IPs parsed to integer")
						df = self.cleaner.map_ips(df, "exaddr")
						dftest = self.cleaner.map_ips(dftest, "exaddr")
						logging.info("EXADDR IPs parsed to integer")
					if conf["cleanData"]["removeFeature"]["enabled"]:
						df = self.cleaner.removeFeatures(df, conf["cleanData"]["removeFeature"]["list"])
						dftest = self.cleaner.removeFeatures(dftest, conf["cleanData"]["removeFeature"]["list"])
					self.set_df(df)
					self.set_dftest(dftest)
				else:
					if conf["cleanData"]["parseIP"]:
						df = self.cleaner.map_ips(df, "srcaddr")
						logging.info("Source IPs parsed to integer")
						df = self.cleaner.map_ips(df, "dstaddr")
						logging.info("Destination IPs parsed to integer")
						#df = self.cleaner.map_ips(df, "nexthop")
						#logging.info("NextHop IPs parsed to integer")
						df = self.cleaner.map_ips(df, "exaddr")
						logging.info("EXADDR IPs parsed to integer")
					if conf["cleanData"]["removeFeature"]["enabled"]:
						df = self.cleaner.removeFeatures(df, conf["cleanData"]["removeFeature"]["list"])
					self.set_df(df)
			self.update_X_y()	

	#TODO actualizar o eliminar
	def analyzeDataset(self):
		conf = self.get_conf_file()
		df = self.get_df()
		if not conf["analyzeData"]["enabled"] or self.cleaner == None:
			logging.error("Analyze option not activated. Change configuration file.")
		else:
			if conf["analyzeData"]["nanValues"]:
				logging.info("Nan search analysis started")
				df = self.cleaner.search_NaN(df)
				logging.info("Nan search analysis completed")
			if conf["analyzeData"]["infiniteValues"]["enabled"]:
				logging.info("Infinite value search analysis started")
				self.cleaner.search_infinite(df, conf["analyzeData"]["infiniteValues"]["list"])
				logging.info("Infinite search analysis completed")
			if conf["analyzeData"]["negativeValues"]:
				logging.info("Negative values search analysis started")
				self.cleaner.search_negatives(df)
				logging.info("Negative values search analysis completed")
			if conf["analyzeData"]["unimportantCol"]:
				logging.info("Unimportant columns analysis started")
				self.cleaner.search_unimportant(df)
				logging.info("Unimportant columns analysis completed")
			if conf["analyzeData"]["export"]:
					self.set_df(df)
					self.to_csv()
					logging.info("Dataset saved and exported to csv format")


	#When pre-processing the data what happens is that the Label column is also being modified, so that everything stops working, also the name of the columns disappears from the csv
	def preprocessDataset(self):
		conf = self.get_conf_file()			
		if conf["preprocess"]["enabled"]:
			preprocess = Preprocessor.Preprocessor()
			logging.info("Preprocessor object created.")
			#TODO Varianza
			#if conf["preprocess"]["variance"]["enabled"]:
				#df = preprocessor.variance(df, conf["preprocess"]["variance"]["limit"], conf["preprocess"]["variance"]["list"])
			#columns,label,df_withOut_label = preprocessor.adapt_dataset(df)
			#We update the dataframe without the label column for preprocessing
			#TODO Mirar a que parte del dataframe afecta
			if conf["preprocess"]["maxScaler"]["enabled"] == True:
				if conf["autotest"]["enabled"] == True:
					new_x, new_x_dftest = preprocess.maxScalerNormalizationAuto(self.get_X(),self.get_X_dftest())
					self.set_X(new_x)
					self.set_X_dftest(new_x_dftest)
				else:
					new_x = preprocess.maxScalerNormalization(self.get_X())
					self.set_X(new_x)
					logging.info("maxScalerNormalization aplied")
			
			if conf["preprocess"]["robustScaler"]["enabled"] == True:
				if conf["autotest"]["enabled"] == True:
					new_x, new_x_dftest = preprocess.robustScalerAuto(self.get_X(),self.get_X_dftest())
					self.set_X(new_x)
					self.set_X_dftest(new_x_dftest)
				else:
					new_x=preprocess.robustScaler(self.get_X())
					self.set_X(new_x)
					logging.info("robustScaler aplied")
					print("Robusto aplicado")				

			if conf["preprocess"]["minScaler"]["enabled"] == True:
				if conf["autotest"]["enabled"] == True:
					new_x, new_x_dftest = preprocess.minScalerNormalizationAuto(self.get_X(),self.get_X_dftest())
					self.set_X(new_x)
					self.set_X_dftest(new_x_dftest)
				else:
					new_x= preprocess.minScalerNormalization(self.get_X())
					self.set_X(new_x)
					logging.info("minScalerNormalization aplied")
					print("minScaler aplicado")	

			if conf["preprocess"]["standardScaler"]["enabled"] == True:
				if conf["autotest"]["enabled"] == True:
					new_x, new_x_dftest = preprocess.standardScalerAuto(self.get_X(),self.get_X_dftest())
					self.set_X(new_x)
					self.set_X_dftest(new_x_dftest)
				else:
					new_x=preprocess.standardScaler(self.get_X())
					self.set_X(new_x)
					logging.info("standardScaler aplied")		

			logging.error("Preprocess option not activated. Change configuration file.")

	#TODO ARREGLAR KBeast
	def reductFeatures(self):
		#self.update_X_y(self.target)			
		conf = self.get_conf_file()
		if conf["featureReduction"]["enabled"]:
			df = self.get_df()
			freductor = FeatureReductor.FeatureReductor()
			if conf["featureReduction"]["kbest"]["enabled"]:
				logging.info("Kbest reduction started")
				function = conf["featureReduction"]["kbest"]["function"]
				kvalue = conf["featureReduction"]["kbest"]["k"]
				oldX = self.get_X()
				oldy = self.get_y()
				new_df = freductor.selectKBest(function, kvalue, oldX, oldy)
				logging.info("Kbest reduction finished")
				new_df["Label"] = oldy
				self.set_df(new_df)
				self.update_X_y(self.target)
				#self.splitDf(self.size,self.seed)
				self.set_updated(True)
			if conf["featureReduction"]["pca"]["enabled"]:
				logging.info("pca reduction stated")
				n_components = conf["featureReduction"]["pca"]["n_components"]
				random_state = conf["featureReduction"]["pca"]["random_state"]
				whiten = conf["featureReduction"]["pca"]["whiten"]
				#Appled PCA
				#Calculate PCA component
				#logging.info("The right number of dimensions for this dataset is:  "+ str(freductor.calculatePCAComponents(self.get_X())))
				#print("The right number of dimensions for this dataset is:  "+ str(freductor.calculatePCAComponents(self.get_X())))
				if conf["autotest"]["enabled"] == True:
					new_x, new_x_dftest = freductor.reductionPCA_auto(self.get_X(), self.get_X_dftest(), n_components, random_state, whiten)
					self.set_X(new_x)
					self.set_X_dftest(new_x_dftest)
				else:
					new_x = freductor.reductionPCA_one(self.get_X(), n_components, random_state, whiten)
					self.set_X(new_x)
				logging.info("pca reduction done")
		else: 
			logging.error("Feature reduction option not activated. Change configuration file.")

	def custom(self, clf, X, y):
		clf = clf.fit(self.X_train, self.y_train)
		y_test_pred = clf.predict(self.X_df_test)
		test = accuracy_score(self.y_df_test , y_test_pred)
		return float(test)

	def votingclasiffier(self, voting_import, voting_name, estimators, voting):
		conf = self.get_conf_file()
		models = []

		for clf in estimators:
			model_import = conf["Models"][clf]["import"]
			model_name = conf["Models"][clf]["name"]

			module = importlib.import_module(model_import)
			function = getattr(module, model_name)

			if "parameters" in conf["Models"][clf]:
				params = conf["Models"][clf]["parameters"]
				model = function(**params)
			else:
				model = function()

			model.modelName = clf
			
			models.append((model_name,model))


		#Create de voting model
		module = importlib.import_module(voting_import)
		function = getattr(module, voting_name)

		model = function(voting=voting, estimators=models)
		model.modelName = voting_name

		return model




	def createModels(self):
		#scorer = make_scorer(self.custom, greater_is_better = True, needs_proba=True)
		conf = self.get_conf_file()
		#Esto lo hacemos porque al reducir la dimenrsionalidad el df cambia y ya no tenemos la columna label, esta la hemos almacenado ya previamente en dicha funcion
		#self.update_X_y(self.target)
		#Split the df
		if conf["export"]["enabled"] == True:
				self.to_csv()
				logging.info("Dataset saved and exported to csv format")

		self.splitDf(self.size,self.seed)
		#We check if all dataset has to be tested
		#if self.data_type != "npy":
		#	self.test_all_dataset()
		#We check whether it is intended to test models or train new ones
		if conf["testSaveModels"]["enabled"]:
			for base, dirs, files in os.walk(conf["models_path"] + self.data_type + "/"):
				for name in files:
					imported_model = load(conf["models_path"] + self.data_type + "/" + name)
					imported_model.modelName = name
					self.testModels(imported_model,self.get_X(),self.get_y())
		else:
			var = conf["Models"]
			#We check if model is enabled and import it 
			for key in sorted(var.keys()):
				if conf["Models"][key]["enabled"] and (key != "Save_Models"):
					model_import = conf["Models"][key]["import"]
					model_name = conf["Models"][key]["name"]

					if key != "Voting_Classifier":


						module = importlib.import_module(model_import)
						function = getattr(module, model_name)
						
						#We check if model imported has parameters
						if "parameters" in conf["Models"][key]:
							params = conf["Models"][key]["parameters"]
							model = function(**params)
						else:
							model = function()
						
						model.modelName = key

					elif key == "Voting_Classifier":

						model = self.votingclasiffier(model_import, model_name, conf["Models"][key]["estimators"], conf["Models"][key]["voting"])

					#We check if the gridsearch option is enabled
					if conf["Models"][key]["GridSearch"]["enabled"]:
						
						clf = GridSearchCV(model, conf["Models"][key]["GridSearch"]["dictionary"], cv=5, verbose=10, n_jobs=4, scoring=self.custom)
						clf.fit(self.get_X(), self.get_y())
						logging.info("Best params: " + str(clf.best_params_))
						#print(clf.best_params_)
						#model_type = conf["Models"][key]["type"]
						#model = test.gridSearch(model, self.get_X(), self.get_y(), conf["Models"][key]["GridSearch"]["dictionary"],model_type)
						model = clf
						model.modelName = key
						logging.info("GridSearch applied to " + model.modelName + " model")
					else:
						test.fit(model, self.get_X_train(), self.get_y_train())

					#We save the models into files and we test it
					if conf["Models"]["Save_Models"]["enabled"]:
						dump(model, conf["models_path"] + self.data_type + "/" + key + '.joblib')
					logging.info("Model " + key + " created")


					#if is novelty, the train dont have results

					if conf["autotest"]["enabled"] == True:
						if conf["Models"][key]["type"]== "novelty":
							print("The Novelty model  " + model.modelName + " was trained fine")
							print("TEST RESULTS : \n\n")
							self.testModels(model,self.get_X_dftest(),self.get_y_dftest())
						else:
							print("TEST RESULTS : \n\n")
							self.testModels(model,self.get_X_dftest(),self.get_y_dftest())
					else:
						if conf["Models"][key]["type"]== "novelty":
							print("The Novelty model  " + model.modelName + " was trained fine")
						else:
							print("TRAINING RESULTS : \n\n")
							self.testModels(model,self.get_X_test(),self.get_y_test())


	#We testing the models created or models saved	
	def testModels(self, model,X_test,y_test):
		conf = self.get_conf_file()
		predictions = test.predict(model, X_test)
		#TODO ESTO HAY QUE BUSCAR UNA FORMA MEJOR....
		if model.modelName == "One_Class_SVM" or model.modelName == "One_Class_SVM.joblib" or model.modelName == "OneClassSVM.joblib"  or model.modelName == "Isolation_Forest" or model.modelName == "Isolation_Forest.joblib" or model.modelName == "IsolationForest.joblib" or model.modelName == "Local_Outlier_Factor" or model.modelName == "Local_Outlier_Factor.joblib" or model.modelName ==  "Local_Outlier_Factor"  or model.modelName == "LocalOutlierFactor.joblib":
			predictions = self.changeAnomalies(predictions)
		print( '\n\n\t\t\t' + model.modelName + '\n\n\n' )
		#print("Params: " + str(model.get_params()) + '\n')
		if conf["Metrics"]["accuracy"]:
			logging.info("Accuracy for model is: %f \n" %  test.accuracyScore(y_test, predictions))
			print("Accuracy for model is: %f \n" %  test.accuracyScore(y_test, predictions))
		if conf["Metrics"]["report"]:
			test.classificationReport(predictions, y_test)
			logging.info("Classification report shown")
		if conf["Metrics"]["confusionMatrix"]:
			cm = test.confusionMatrix(predictions, y_test, ["Attack", "Not attack"])
			print(cm)
			logging.info("Confusion Matrix")
			print('\n\n')
			if conf["Metrics"]["plotConfusionMatrix"]:
				test.plotConfusionMatrix(model.modelName,cm,["", ""])
				logging.info("Confusion Matrix plotted")
		if conf["Metrics"]["cohenKappaScore"]:
			test.cohenKappaScore(predictions, y_test)
			logging.info("Cohen Kappa Score report shown")
		if conf["Metrics"]["matthewsCorrcoef"]:
			test.matthewsCorrcoef(predictions, y_test)
			logging.info("Matthews Corrcoef report shown")
		if conf["Metrics"]["rocCurve"]:
				test.plotRocCurve(model,X_test, y_test)
				logging.info("Confusion Matrix plotted")
		if conf["Metrics"]["overfitting"]:
				#TODO REVISAR ESTA METRICA ANTES SE USABA XTRAIN YTRAIN
		        test.learning_curves(model,X_test, y_test)
		        logging.info("Learning curves plotted")

		print("##########################################################")
			
	def changeAnomalies(self,predictions):
		for i in range(len(predictions)):
			if predictions[i] == 1:
				#print("normal")
				predictions[i] = 0
			if predictions[i] == -1:
				#print("anomalia")
				predictions[i] = 1
		return predictions

	


	#Getters---------------------------------------------------------------------	
	
	def get_df(self):
		return self.df

	def get_X(self):
		return self.X

	def get_y(self):
		return self.y
	
	def get_X_dftest(self):
		return self.X_df_test

	def get_y_dftest(self):
		return self.y_df_test
	
	def get_X_train(self):
		return self.X_train
	
	def get_y_train(self):
		return self.y_train
	
	def get_X_test(self):
		return self.X_test
	
	def get_y_test(self):
		return self.y_test

	def get_conf_file(self):
		return self.conf_file

	def get_cleaner(self):
		return self.cleaner

	def get_updated(self):
		return self.updated

	def get_dftest(self):
		return self.test_df

	#Setters---------------------------------------------------------------------

	def set_df(self, df):
		self.df = df
		logging.info("Atributte df of instance of class MoEv changed")

	def set_dftest(self, df):
		self.test_df = df


	def set_X_dftest(self, X):
		self.X_df_test = X
		logging.info("Atributte X of instance of class MoEv changed")

	def set_y_dftest(self, y):
		self.y_df_test = y
		logging.info("Atributte y of instance of class MoEv changed")

	def set_X(self, X):
		self.X = X
		logging.info("Atributte X of instance of class MoEv changed")

	def set_y(self, y):
		self.y = y
		logging.info("Atributte y of instance of class MoEv changed")
	
	def set_X_train(self, X_train):
		self.X_train = X_train
		logging.info("Atributte X_train of instance of class MoEv changed")
	
	def set_y_train(self, y_train):
		self.y_train = y_train
		logging.info("Atributte y_train of instance of class MoEv changed")
	
	def set_X_test(self, X_test):
		self.X_test = X_test
		logging.info("Atributte X_test of instance of class MoEv changed")
	
	def set_y_test(self, y_test):
		self.y_test = y_test
		logging.info("Atributte y_test of instance of class MoEv changed")

	def set_conf_file(self, file):
		self.conf_file = file
		logging.info("Atributte conf_file of instance of class MoEv changed")

	def set_updated(self,updated ):
		self.updated = updated

