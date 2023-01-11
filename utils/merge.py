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
import argparse

#Parser to interpret command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help="Directory path with the CSVs to merge")
parser.add_argument("-o", "--output", help="Destination csv file path")
args = parser.parse_args()

if args.directory:
	os.chdir(args.directory)
else:
	print("Error. A directory with csv files is needed")
	parser.print_help()
	quit()

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], copy=False)

#export to csv
if args.output:
	output = args.output
else:
	output = "csv_combined.csv"

combined_csv.to_csv(output, index=False, encoding='utf-8-sig')



