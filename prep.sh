##!/bin/sh

mkdir -p datasets
cd datasets
pwd

#PAMAP
#wget http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip
#unzip PAMAP2_Dataset.zip
#python ../convert_data_to_np.py PAMAP

#UCI
mkdir -p UCI2
cd UCI2
pwd
#wget http://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip
#unzip HAPT\ Data\ Set.zip
cd ..
pwd
#python ../convert_data_to_np.py UCI-raw

#WISDM-watch
#wget https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip
#unzip wisdm-dataset.zip
#python ../convert_data_to_np.py WISDM-watch

#mkdir -p capture24
#cd capture24/

#for i in $(seq -w 151)
#do
#    curl -JLO "https://ora.ox.ac.uk/objects/uuid:92650814-a209-4607-9fb5-921eab761c11/download_file?safe_filename=P${i}.csv.gz&type_of_work=Dataset"
#done
#
#curl -JLO "https://ora.ox.ac.uk/objects/uuid:92650814-a209-4607-9fb5-921eab761c11/download_file?safe_filename=metadata.csv&type_of_work=Dataset"
#curl -JLO "https://ora.ox.ac.uk/objects/uuid:92650814-a209-4607-9fb5-921eab761c11/download_file?safe_filename=annotation-label-dictionary.csv&type_of_work=Dataset"
#
#
#for f in $(ls); do 
#    if [ ${f: -2} == "gz" ]; then 
#        gunzip $f; 
#    fi; 
#done

#WISDM-v1
wget https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
gunzip WISDM_ar_latest.tar.gz
tar -xf WISDM_ar_latest.tar

python ../convert_data_to_np.py WISDM-v1

mkdir -p realdisp
cd realdisp
mkdir -p RawData
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00305/realistic_sensor_displacement.zip
unzip realistic_sensor_displacement.zip
cd ..
python ../convert_data_to_np.py REALDISP
cd ..
pwd
