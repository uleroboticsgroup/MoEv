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
import argparse
import datetime
import ipaddress

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Source file")
parser.add_argument("-r", "--remove",nargs='?',const='remove', help="Remove traffic")
parser.add_argument("-p", "--prot",nargs='*', help="Protocol to remove")
parser.add_argument("-dp", "--dstport",nargs='*', help="Destination Port to remove")
parser.add_argument("-dip", "--dstip",nargs='*', help="Destiantion Ip to remove")
parser.add_argument("-sip", "--srcip",nargs='*', help="Source Ip to remove")
parser.add_argument("-l", "--label", help="Label attack | normal")
parser.add_argument("-o", "--output", help="output")
args = parser.parse_args()

dstips = []
srcips = []
ports = []
protocols = []

if not args.file:
	print("A csv file is required")
	parser.print_help()
	quit()

dataset = pd.read_csv(args.file)

if not args.label:
	print("A label is required")
	parser.print_help()
	quit()
if args.label == 'attack':
	dataset["Label"] = 1

if args.label == 'normal':
	dataset["Label"] = 0


if args.remove:

	if args.dstip:
		print(args.dstip)

		try:
			for IP in args.dstip:
				ip = str(ipaddress.ip_address(IP))
				dstips.append(ip)
		except ValueError:
			print("Ip format invalid")
			parser.print_help()
			quit()

	if args.srcip:
		print(args.srcip)

		try:
			for IP in args.srcip:
				ip = str(ipaddress.ip_address(IP))
				srcips.append(ip)
		except ValueError:
			print("Ip format invalid")
			parser.print_help()
			quit()

	if args.prot:
		print(args.prot)

		for prot in args.prot:
			protocols.append(int(prot))

	if args.dstport:
		print(args.dstport)

		for port in args.dstport:
			ports.append(int(port))


	if args.prot and args.dstport and args.dstip and args.srcip:
		index = dataset.loc[(dataset['prot'].isin(protocols)) & (dataset['dstport'].isin(ports)) & (dataset['dstaddr'].isin(dstips)) & (dataset['srcaddr'].isin(srcips))].index.values.astype(int)
		dataset = dataset.drop(index)
	if args.prot and args.dstport and args.dstip and not args.srcip:
		index = dataset.loc[(dataset['prot'].isin(protocols)) & (dataset['dstport'].isin(ports)) & (dataset['dstaddr'].isin(dstips))].index.values.astype(int)
		dataset = dataset.drop(index)
	if args.prot and args.dstport and not args.dstip and args.srcip:
		index = dataset.loc[(dataset['prot'].isin(protocols)) & (dataset['dstport'].isin(ports)) & (dataset['srcaddr'].isin(srcips))].index.values.astype(int)
		dataset = dataset.drop(index)
	if args.prot and args.dstport and not args.dstip and not args.srcip:
		index = dataset.loc[(dataset['prot'].isin(protocols)) & (dataset['dstport'].isin(ports))].index.values.astype(int)
		dataset = dataset.drop(index)
	if args.prot and not args.dstport and args.dstip and args.srcip:
		index = dataset.loc[(dataset['prot'].isin(protocols)) & (dataset['dstaddr'].isin(dstips)) & (dataset['srcaddr'].isin(srcips))].index.values.astype(int)
		dataset = dataset.drop(index)
	if args.prot and not args.dstport and args.dstip and not args.srcip:
		index = dataset.loc[(dataset['prot'].isin(protocols)) & (dataset['dstaddr'].isin(dstips))].index.values.astype(int)
		dataset = dataset.drop(index)
	if args.prot and not args.dstport and not args.dstip and args.srcip:
		index = dataset.loc[(dataset['prot'].isin(protocols)) & (dataset['srcaddr'].isin(srcips))].index.values.astype(int)
		dataset = dataset.drop(index)
	if args.prot and not args.dstport and not args.dstip and not args.srcip:
		index = dataset.loc[(dataset['prot'].isin(protocols))].index.values.astype(int)
		dataset = dataset.drop(index)




 
	if not args.prot and args.dstport and args.dstip and args.srcip:
		index = dataset.loc[(dataset['dstport'].isin(ports)) & (dataset['dstaddr'].isin(dstips)) & (dataset['srcaddr'].isin(srcips))].index.values.astype(int)
		dataset = dataset.drop(index)
	if not args.prot and args.dstport and args.dstip and not args.srcip:
		index = dataset.loc[(dataset['dstport'].isin(ports)) & (dataset['dstaddr'].isin(dstips))].index.values.astype(int)
		dataset = dataset.drop(index)
	if not args.prot and args.dstport and not args.dstip and args.srcip:
		index = dataset.loc[(dataset['dstport'].isin(ports)) & (dataset['srcaddr'].isin(srcips))].index.values.astype(int)
		dataset = dataset.drop(index)
	if not args.prot and args.dstport and not args.dstip and not args.srcip:
		index = dataset.loc[dataset['dstport'].isin(ports)].index.values.astype(int)
		dataset = dataset.drop(index)


	if not args.prot and not args.dstport and args.dstip and args.srcip:
		index = dataset.loc[(dataset['dstaddr'].isin(dstips)) & (dataset['srcaddr'].isin(srcips))].index.values.astype(int)
		dataset = dataset.drop(index)
	if not args.prot and not args.dstport and args.dstip and not args.srcip:
		index = dataset.loc[(dataset['dstaddr'].isin(dstips))].index.values.astype(int)
		dataset = dataset.drop(index)



	if not args.prot and not args.dstport and not args.dstip and args.srcip:
		index = dataset.loc[(dataset['srcaddr'].isin(srcips))].index.values.astype(int)
		dataset = dataset.drop(index)

if args.srcip:
	if not args.remove:
		print("Argument remove is requiered")
		parser.print_help()
		quit()

if args.dstip:
	if not args.remove:
		print("Argument remove is requiered")
		parser.print_help()
		quit()

if args.prot:
	if not args.remove:
		print("Argument remove is requiered")
		parser.print_help()
		quit()

if args.dstport:
	if not args.remove:
		print("Argument remove is requiered")
		parser.print_help()
		quit()



if args.output:
	csv_output = args.output
else:
	csv_output = "./csv_labelled.csv"""

dataset.to_csv(csv_output, index=False, encoding='utf-8-sig')
