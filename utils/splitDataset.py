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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Archivo origen")
parser.add_argument("-p1", "--percentage1", help="First subdataset percentage")
parser.add_argument("-p2", "--percentage2", help="Second subdataset percentage")
parser.add_argument("-b1", "--balance1", help="New subdataset 1 Bening values percentage")
parser.add_argument("-b2", "--balance2", help="New subdataset 2 Bening values percentage")
parser.add_argument("-o1", "--output1", help="First subdataset")
parser.add_argument("-o2", "--output2", help="Second subdataset")
args = parser.parse_args()

if not args.file:
	print("A csv file is required")
	parser.print_help()
	quit()

if not args.output1:
    output1 = "dataset1.csv"
else:
    output1 = args.output1

if not args.output2:
    output2 = "dataset2.csv"
else:
    output2 = args.output2

input_file = args.file

p1 = int(args.percentage1)
p2 = int(args.percentage2)
b1 = int(args.balance1)
b2 = int(args.balance2)

if p1 < 0 and p1 > 100 or p2 < 0 and p2 > 100 or b1 < 0 and b1 > 100 or b2 < 0 and b2 > 100:
    print("Error with percentages. Please introduce values between 1 and 100.")
    quit()


print("Starting process...")

df = pd.read_csv(input_file)

#Separate the rows that are labeled as attack from the benign
bening_list = df.loc[df["Label"] == 0]
attack_list = df.loc[df["Label"] != 0]

original_size = len(df)

#We calculate the size of the new datasets according to the percentage entered by the user
new_size1 = (p1 * original_size) // 100
new_size2 = (p2 * original_size) // 100

#We calculate the number of rows of each traffic according to the percentage entered by the user
bening_size1 = (b1 * new_size1) // 100
bening_size2 = (b2 * new_size2) // 100

attack_size1 = new_size1 - bening_size1
attack_size2 = new_size2 - bening_size2
#---------------------------------------------------------------------------------------------

print("The original dataset size is:%i" % original_size)
print("The dataset1 size is:%i with bening values:%i and attack values:%i" % (new_size1, bening_size1, attack_size1))
print("The dataset2 size is:%i with bening values:%i and attack values:%i" % (new_size2, bening_size2, attack_size2))

#We select the rows of the new datasets randomly
bening1 = bening_list.sample(n = bening_size1)
attack1 = attack_list.sample(n = attack_size1)

bening2 = bening_list.sample(n = bening_size2)
attack2 = attack_list.sample(n = attack_size2)
#-----------------------------------------------------------------

# We build an array with the datasets of different traffic
frames1 = [bening1, attack1] 
frames2 = [bening2, attack2] 

#We merge the subsets
dataset1 = pd.concat(frames1)
dataset2 = pd.concat(frames2) 

dataset1.to_csv(output1, index=False, encoding='utf-8-sig')
dataset2.to_csv(output2, index=False, encoding='utf-8-sig')



