import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

if __name__ == "__main__":
    df = pd.read_csv('training_data.csv')
    result = df.groupby('Murmur')['Patient ID'].agg(list).reset_index()

    array_Absent = result.loc[result['Murmur'] == 'Absent', 'Patient ID'].values[0]
    array_Present = result.loc[result['Murmur'] == 'Present', 'Patient ID'].values[0]
    array_Unknown = result.loc[result['Murmur'] == 'Unknown', 'Patient ID'].values[0]

    result = df.groupby('Outcome')['Patient ID'].agg(list).reset_index()
    array_Abnormal = result.loc[result['Outcome'] == 'Abnormal', 'Patient ID'].values[0]
    array_Normal = result.loc[result['Outcome'] == 'Normal', 'Patient ID'].values[0]
    array1 = array_Abnormal
    array = [array_Absent,array_Present,array_Unknown]

    array_out_Abnormal = [[],[],[]]
    i = 0
    for array2 in array:
        array_out_Abnormal[i] = np.intersect1d(array1, array2)
        i = i+1
        
    array1 = array_Normal
    array = [array_Absent,array_Present,array_Unknown]

    array_out_Normal = [[],[],[]]
    i = 0
    for array2 in array:
        array_out_Normal[i] = np.intersect1d(array1, array2)
        i = i+1
    x_test =[]
    for i in range(len(array_out_Abnormal)):
        if i == 0 :
            size =  70
        elif i == 1:
            size = 18
        elif i ==2:
            size =7    
        sampled_array = np.random.choice(array_out_Abnormal[i], size=size, replace=False)
        x_test.extend(sampled_array)
    for i in range(len(array_out_Normal)):
        if i == 0 :
            size =  70
        elif i == 1:
            size = 18
        elif i ==2:
            size =7    
        sampled_array = np.random.choice(array_out_Normal[i], size=size, replace=False)
        x_test.extend(sampled_array)
    id_train = np.setdiff1d(df['Patient ID'], x_test)
    file = open('id_patient.csv','wb')
    pickle.dump((id_train,x_test),file)
    file.close()
    file = open('id_patient.csv','rb')
    data = pickle.load(file)
    id_train,id_test = data
    file.close()
    def time_start(filename):
        file_tsv = filename[:-4]+".tsv"
        df_time = pd.read_csv(str(file_tsv))
        df_time = np.array(df_time)
        # print(len(df_time))
        # print(df[0][0].split('\t')[0])
        # print(df[len(df)-1][0].split('\t')[0])
        start = (df_time[0][0].split('\t')[0])
        end = (df_time[len(df_time)-1][0].split('\t')[0])
        return start+"_"+end

    data = 'the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data'

    out_file = 'training_data_all/'
    df = pd.read_csv('the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data.csv')
    # Danh sách các thư mục cần tạo
    folders = [
        "training_data_all/train/Absent",
        "training_data_all/train/Present",
        "training_data_all/train/Unknown",
        "training_data_all/test/Absent",
        "training_data_all/test/Present",
        "training_data_all/test/Unknown",
        "training_data_all/val/Absent",
        "training_data_all/val/Present",
        "training_data_all/val/Unknown"
    ]

    # Xóa các thư mục cũ và tạo lại
    for folder in folders:
        if os.path.exists(folder):
            # Xóa thư mục và tất cả nội dung bên trong
            shutil.rmtree(folder)

            print(f"Đã xóa thư mục: {folder}")
        
        # Tạo lại thư mục
        os.makedirs(folder)

        print(f"Đã tạo thư mục mới: {folder}")
    
    for element in id_test:
        for filename in os.listdir(data):
            if filename.startswith(str(element)) and filename.endswith('.wav'):
                link_file = os.path.join(data,filename)
                time = (time_start(link_file))
                murmur_value = df.loc[df['Patient ID'] == element, 'Murmur'].values[0]
                outcome_value = df.loc[df['Patient ID'] == element, 'Outcome'].values[0]
                shutil.copy(link_file, out_file +'test/'+murmur_value+'/'+outcome_value+'_'+str(time)+'_'+filename)
                
    for element in id_train:
        for filename in os.listdir(data):
            if filename.startswith(str(element)) and filename.endswith('.wav'):
                link_file = os.path.join(data,filename)
                time = (time_start(link_file))
                murmur_value = df.loc[df['Patient ID'] == element, 'Murmur'].values[0]
                outcome_value = df.loc[df['Patient ID'] == element, 'Outcome'].values[0]
                shutil.copy(link_file, out_file +'train/'+murmur_value+'/'+outcome_value+'_'+str(time)+'_'+filename)

    