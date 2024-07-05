#!/bin/sh
read -p "input batch_size :" batch
read -p "input for number of epcochs :" epochs
read -p "input the values of layers :" layers
read -p "number of folds :" folds
names=$(ls datasets/validation/| grep validation.txt | cut -d  "-" -f 1) 
list=()
for nam in $names;do 
      s=datasets/validation/$nam"-validation.txt"
      r=datasets/$nam"-balanced.csv"
      if [[ $layers == *","* ]]; then
             lay1=$(echo $layers|cut -d "," -f 1)
             lay2=$(echo $layers|cut -d "," -f 2)
             a=$(head -n 1 $s| cut -d " " -f 5)
             pipenv run python3 main.py  -i $r -o $nam --num_samples_class_benign $a  --num_samples_class_malware $a --batch_size $batch  --number_epochs $epochs  --dense_layer_sizes_g $lay1,$lay2 --dense_layer_sizes_d $lay2,$lay1 --k_fold $folds 
      else
          a=$(head -n 1 $s| cut -d " " -f 5)
          pipenv run python3 main.py  -i $r -o $nam --num_samples_class_benign $a  --num_samples_class_malware $a --batch_size $batch  --number_epochs $epochs  --dense_layer_sizes_g $layers --dense_layer_sizes_d $layers  --k_fold $folds
      fi
      echo "executado"
      #list+=($a)
      
      
done
#for i in ${list[*]};do 
 #	pipenv run python3 main_aim_mlflow.py -ml true -i r
#done
echo ${list[*]}



