import json
import os
import sys
import datetime
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def create_argparse(): 
          parser=argparse.ArgumentParser()
          parser.add_argument('-i','--input_file',type=str,required=True,default=None ,help='Input file contaning thew samples')
          parser.add_argument("-o","--output_file",type=str,required=True,default=None, help="Name of the output file for saving the results")
          return  parser.parse_args()

def validation(arguments):
   if arguments.input_file !=None:
      samples_file=pd.read_csv(arguments.input_file)
      count=0
      f=open(arguments.output_file,"w")
      if samples_file.empty:
      		print("empty dataset",file=f)
      		return 1
      columns=samples_file.columns.values.tolist()
      list_of_columns_missing_values=[]
      list_of_columns_with_wrong_values=[]
      print("yes")
      
      for values in columns:
                 if samples_file[values].isnull().any() ==True:
                 	e=samples_file[sample_file[values].isnull()].index.tolist()
                 	e.append(values)
                 	list_of_columns_missing_values.append(e)
                 if samples_file[values].isin([0, 1]).all()==False:
                        e=samples_file[(samples_file[values] != 0) & (samples_file[values] != 1)].index.tolist()
                        e.append(values)
                        list_of_columns_with_wrong_values.append(e)
      if list_of_columns_missing_values:
      		for column in list_of_columns_missing_values:
      			print("a coluna ",column[-1], "possui as seguintes entradas vazias",column[:-1],file=f )
      if list_of_columns_with_wrong_values:
               for column in list_of_columns_with_wrong_values:
                      print("a coluna ",column[-1], "possui as seguintes entradas não iguais aos valores esperados",file=f)			
      malware_samples=samples_file['class'].value_counts()[1]
      benign_samples=samples_file['class'].value_counts()[0]
      print("number of malware samples",malware_samples,file=f)
      print("numbers of benign samples",benign_samples,file=f)
      print("número de colunas",len(samples_file.columns),"número de linhas",len(samples_file),file=f)
      f.close()
     

 




if __name__ == "__main__":
   arguments=create_argparse()

   validation(arguments)
