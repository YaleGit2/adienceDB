#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:55:46 2020

@author: yalewkidane
"""
from torch.utils.data import Dataset, DataLoader

import pandas as pd
#from random import shuffle
import csv
import numpy as np

from skimage import io #,transform
import matplotlib.pyplot as plt
#from scipy.misc import imresize #,imsave
from PIL import Image
import torch
from torchvision import transforms

from sklearn.model_selection import train_test_split

#Global Variables
Image_Size = 256

DataDirectory="/home/yalegtx/Documents/Project/Adience/Data/adiencedb"
class AdienceDataset(Dataset):
    def __init__(self, imageData, directory_path, transform=None):
        self.imageData_user_id=imageData[:,0] 
        self.imageData_original_image=imageData[:,1] 
        self.imageData_face_id=imageData[:,2]
        self.imageData_age=imageData[:,3]
        self.length = self.imageData_user_id.shape[0]
        self.directory_path = directory_path
        self.transform = transform
    def __len__(self):
        #print("")
        return self.length
        
        
    def __getitem__(self, idx):
        image_file_name="/landmark_aligned_face."+str(self.imageData_face_id[idx]) \
            +"."+ self.imageData_original_image[idx]
        imgade_file_path =self.directory_path+"aligned/"+ \
            self.imageData_user_id[idx]+image_file_name
            
        
        #image = io.imread(imgade_file_path)
        #image = imresize(image, [self.image_size, self.image_size])
        
        image = Image.open(imgade_file_path).convert("RGB")
        #image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
            
        age = self.get_age(self.imageData_age[idx])
        
        sample = {'image': image, 'age': age}
        return sample
    
    #def crop_image(self, input_image_size, output_image_Size):
        
    def get_age(self, age_range):
        #print("-------XXX-----",age_range)
        age=0
        #-----------------------------------------[0-2]
        if age_range=='(0, 2)':
            age = 0
        elif age_range=='2':
            age = 0
        elif age_range=='3':
            age = 0
        #-----------------------------------------[0-2]
        elif age_range=='(4, 6)':
            age = 1
        #-----------------------------------------[8-13]
        elif age_range=='(8, 12)':
            age = 2
        elif age_range=='13':
            age = 2
        #-----------------------------------------[15-22]
        elif age_range=='(15, 20)':
            age = 3
        elif age_range=='22':
            age = 3
        #-----------------------------------------[23-32]
        elif age_range=='23':
            age = 4
        elif age_range=='(25, 32)':
            age = 4
        elif age_range=='(27, 32)':
            age = 4
        elif age_range=='(27, 32)':
            age = 4
        elif age_range=='29':
            age = 4
        elif age_range=='32':
            age = 4
        elif age_range=='34':
            age = 4
        elif age_range=='35':
            age = 4
        #-----------------------------------------[38-45]
        elif age_range=='36':
            age = 5
        elif age_range=='(38, 43)':
            age = 5
        elif age_range=='(38, 42)':
            age = 5
        elif age_range=='42':
            age = 5
        elif age_range=='45':
            age = 5
        elif age_range=='46':
            age = 5
        elif age_range=='(38, 48)':
            age = 5
        #-----------------------------------------[48-55]
        elif age_range=='(48, 53)':
            age = 6
        elif age_range=='55':
            age = 6
        elif age_range=='56':
            age = 6
        #-----------------------------------------[57-100]
        elif age_range=='57':
            age = 7
        elif age_range=='58':
            age = 7
        elif age_range=='(60, 100)':
            age = 7
        else:
            age = age_range
        return age


def show_image_from_tensor(image):
    img = image.permute(1,2,0)
    plt.imshow(img)
    plt.show()

def main_1(directory_path):
    imageData_csv = pd.read_csv(
        directory_path+"/Z-CSV-Files/combined_csv.csv", sep=',')
    train, test = train_test_split(imageData_csv.values, test_size=0.2)
    print(train.shape)
    print(test.shape)
    
    train_dataset = AdienceDataset(
            imageData=train, 
            directory_path=directory_path,
            image_size = Image_Size)
    
    for i in range (len(train_dataset)):
        sample = train_dataset[i]
        sample_age=np.array(sample['age'])
        #print("-------------",sample_age)
        age =0
        if sample_age==0:
            age = 0
        elif sample_age==1:
            age = 1
        elif sample_age==2:
            age = 2
        elif sample_age==3:
            age = 3
        elif sample_age==4:
            age = 4
        elif sample_age==5:
            age = 5
        elif sample_age==6:
            age = 6
        elif sample_age==7:
            age = 7
        else:
            print('sample_batched_age : ',sample_age)
            
        
        
    return age


def main(directory_path):
    imageData_csv = pd.read_csv(
        directory_path+"/Z-CSV-Files/combined_csv.csv", sep=',')
    train, test = train_test_split(imageData_csv.values, test_size=0.2)
    print(train.shape)
    print(test.shape)
    composed = transforms.Compose([#transforms.ToPILImage(),
                                   transforms.Resize(256),
                                   transforms.CenterCrop(227),
                                   transforms.ToTensor()])
    
    train_dataset = AdienceDataset(
            imageData=train, 
            directory_path=directory_path,
            transform = composed)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)
    '''
    test_dataset = AdienceDataset(
            imageData=test, 
            directory_path=directory_path),
            image_size = Image_Size)
    
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=32,
                              shuffle=True,
                             num_workers=2)
    '''
    for epoch in range(1):
        age = [0,0,0,0,0,0,0,0,0]
        print(age)
        for i_batch, sample_batched in enumerate(train_loader, 0):
            #print("-------------21")
            sample_batched_age=np.array(sample_batched['age'])
            #print("-------------",sample_batched_age)
            sample_image=np.array(sample_batched['image'])
            '''
            print('age_batch_size=', sample_batched_age.shape, )
            
            print(" image_bach_size=",sample_batched_image.shape)
            
            print(sample_batched_age)
            plt.imshow(sample_batched_image[0])
            plt.show()
            
            s=input()
            if(s=="exit"):
                break
            
            
            if i_batch == 3:
                break
            
            
            for sample_age in sample_batched_age:
            
                if sample_age==0:
                    age[0] +=1
                elif sample_age==1:
                    age[1] +=1
                elif sample_age==2:
                    age[2] +=1
                elif sample_age==3:
                    age[3] +=1
                elif sample_age==4:
                    age[4] +=1
                elif sample_age==5:
                    age[5] +=1
                elif sample_age==6:
                    age[6] +=1
                elif sample_age==7:
                    age[7] +=1
                else:
                    print('sample_batched_age : ',sample_batched_age, age)
                    age[8] +=1
            
            '''
            print("-----here 0")
            print(sample_image.shape)
            #print(sample_image.type())
            #image=transforms.ToPILImage()(sample_image[0])
            #image = transforms
            sample_image = torch.tensor(sample_image)
            print("torch.tensor", sample_image.shape)
            #real_imgs = sample_image.permute(0,2,3,1)
            #plt.imshow(real_imgs[0])
            #image.show()
            print("-----here 1")
            show_image_from_tensor(sample_image[0])
            print("-----here 3")
            if i_batch == 3:
               break
         
        
        
        
        
if __name__ == '__main__':
    directory_path ='/home/yalegtx/Documents/Project/Adience/Data/adiencedb/'
    #test2()
    #prepare_csv_file()
    #image_test(directory_path)
    main(directory_path)
    #main_1(directory_path)
    
  
        
#----------------------------prepare CSV file----------------------------------        
#csv is seprated by 
# it uses --> import csv
def text_to_csv(text_file_path, csv_file_path):
    txt_file = r''+text_file_path
    csv_file = r''+csv_file_path
    in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
    out_csv = csv.writer(open(csv_file, 'w'))
    
    out_csv.writerows(in_txt)
    
    
def merge_csv_files(file_list, destination):
    imageData = pd.concat([pd.read_csv(file) for file in file_list])
    print('combined_csv: ', imageData.shape)
    imageData.drop(imageData[imageData.age == "None"].index, inplace=True)
    imageData.to_csv( destination, index=False, encoding='utf-8-sig')
    print('combined_csv: ', imageData.shape)

def read_csv(directory_path):
    for i in range(5):
        imageData_0 = pd.read_csv(directory_path+ '/Z-CSV-Files/fold_'+str(i)+'_data.csv' , sep=',')
        print(str(i)+'_data.csv: ', imageData_0.shape)

def prepare_csv_file():
    directory_path ='/home/yalegtx/Documents/Project/Adience/Data/adiencedb/'
    n_list=5
    for i in range(n_list):
        text_to_csv(directory_path+'fold_'+str(i)+'_data.txt', 
                directory_path+ '/Z-CSV-Files/fold_'+str(i)+'_data.csv' )
    
    file_list=[]
    for i in range(n_list):
        file_list.append(directory_path+ '/Z-CSV-Files/fold_'+str(i)+'_data.csv')
    
    #print the shape
    read_csv(directory_path)
    
    #merge the CSVs
    destination = directory_path+"/Z-CSV-Files/combined_csv.csv"
    merge_csv_files( file_list, destination)
    
def test2():
    imageData_0 = pd.read_csv(DataDirectory+'/fold_0_data.csv', sep=',')
    imageData_1 = pd.read_csv(DataDirectory+'/fold_1_data.csv', sep=',')
    frames =[imageData_0,imageData_1]
    imageData = pd.concat(frames)
    imageData.to_csv( DataDirectory+"/combined_csv.csv", index=False, encoding='utf-8-sig')
    print(type(imageData))
    print(imageData.shape)
    print(imageData.columns)
    
    imageData_c0=imageData.loc[:,"user_id"]
    #imageData_c1=imageData.loc[n,1]
    #imageData_c2=imageData.loc[n,3]
    #imageData_c3=imageData.loc[n,4]
    #imageData_c4=imageData.loc[n,5]
    #imageData_c0=shuffle_data(imageData_c0)
    for x in range(5):
        print(imageData_c0[x])
        #print(imageData_c1[x])
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-------------------------show image and test--------------------------------------------

def image_show(image_path):
    image=io.imread(image_path)
    print('image Shape', image.shape)
    plt.imshow(image)
    plt.show()

#def get_file_name():
    

def image_test(directory_path):
    imageData = pd.read_csv(
        directory_path+"/Z-CSV-Files/combined_csv.csv", sep=',')
    print(imageData.shape)
    print(imageData.columns)
    imageData_user_id=imageData.loc[:,"user_id"]
    imageData_original_image=imageData.loc[:,"original_image"]
    imageData_face_id=imageData.loc[:,"face_id"]
    
    print(imageData_user_id.shape)
    
    length = imageData_user_id.shape[0]
    print(length)
    
    image_file_name="/landmark_aligned_face."+str(imageData_face_id[1])+"."+\
        imageData_original_image[1]
    imgade_file_path =directory_path+"aligned/"+imageData_user_id[1]+\
        image_file_name
    
    print(imgade_file_path)
    image_show(imgade_file_path)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
