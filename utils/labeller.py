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


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Source file")
parser.add_argument("-ips", "--srcip", help="Source IP")
parser.add_argument("-ipd", "--dstip", help="Destiantion IP")
parser.add_argument("-e", "--label", help="Label")
parser.add_argument("-pt", "--port", help="Destination port")
parser.add_argument("-p", "--protocol", help="Protocol 6=udp | 17=tcp")
parser.add_argument("-o", "--output", help="Destiantion file")
args = parser.parse_args()

if not args.file:
	print("It is necessary to indicate a file type csv")
	parser.print_help()
	quit()

dataset = pd.read_csv(args.file)

if not args.srcip:
	print("Es necesario indicar una direccion IP de origen")
	parser.print_help()
	quit()
if not args.dstip:
	print("It is necessary to indicate a source IP address")
	parser.print_help()
	quit()
if not args.port:
	print("It is necessary to indicate a destination port")
	parser.print_help()
	quit()
if not args.protocol:
	print("It is necessary to indicate a protocol")
	parser.print_help()
	quit()

index = dataset.loc[(dataset['Src IP'] == args.srcip) & (dataset['Dst IP'] == args.dstip) & (dataset['Dst Port'] == int(args.port)) & (dataset['Protocol'] == 
	int(args.protocol))].index.values.astype(int)

if not args.label:
	print("It is necessary to indicate the desired label")
	parser.print_help()
	quit()

if dataset.iloc[1, -1] == "No Label":
	dataset["Label"].replace("No Label", value="0", inplace=True)

label = args.label

for i in index: 
	dataset.set_value(i, "Label", label)


if args.output:
	csv_output = args.output
else:
	csv_output = "./csv_labelled.csv"""

dataset.to_csv(csv_output, index=False, encoding='utf-8-sig')
