mkdir -p datasets
cd datasets
pwd
#wget https://archive.ics.uci.edu/ml/machine-learning-databases/00227/PAMAP2_Dataset.zip
#unzip PAMAP2_Dataset.zip
#python ../convert_data_to_np.py PAMAP
#mkdir -p UCI2
#cd UCI2
#pwd
#wget http://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip
#unzip HAPT\ Data\ Set.zip
#cd ..
#pwd
#python ../convert_data_to_np.py UCI-raw
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip
unzip ../wisdm-dataset.zip
python ../convert_data_to_np.py WISDM-watch
#wget https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
#gunzip WISDM_ar_latest.tar.gz
#tar -xf WISDM_ar_latest.tar
#python ../convert_data_to_np.py WISDM-v1
cd ..
pwd
