from env import CuttingStock
import os
import pandas as pd
import random

import sys

sys.path.append('.')


# global read_optimal_solutions
# def read_optimal_solutions():
#     path = 'given_solutions/solutions.xlsx'
#     optimal_solutions = pd.read_excel(path)
#     return optimal_solutions

# read file names
def instance_name(name_file):
    # print("name_file : ", name_file)
    # F = open("/content/FileName.txt", "r")
    F = open(name_file, "r")
    file_name = F.read()

    # Files = pd.read_csv('/content/FileName.txt',header =None)
    Files = pd.read_csv(name_file, header=None)
    Files = Files[0].tolist()

    if '.DS_Store' in Files:
        Files.remove('.DS_Store')

    return Files


# read instance given its number
def instance_train(count, name_file, **parameters):
    name = instance_name(name_file)
    # print("instance_test")
    # print("parameters: ", parameters['parameters'])
    # print("*"*100)
    try:
        # data = pd.read_csv('/content/drive/MyDrive/MIE1666/Test/'+name[count],delimiter = "\t",error_bad_lines=False, header = None,skiprows=[0,1])
        # data1 = pd.read_csv('/content/drive/MyDrive/MIE1666/Test/'+name[count],header = None)

        data = pd.read_csv('inputs/instances/Scheduled_train/' + name[count], delimiter="\t", header=None, skiprows=[0, 1])
        data1 = pd.read_csv('inputs/instances/Scheduled_train/' + name[count], header=None)
    except:
        return "not found"

    roll_count = int(data1.iloc[0])
    roll_length = int(data1.iloc[1])
    order_length = data[0].tolist()
    order_count = data[1].tolist()
    name_ = name[count]
    # print(roll_count)
    INSTANCE = CuttingStock(order_count, order_length, roll_length, name_, parameters=parameters['parameters'])
    return INSTANCE


def instance_test(count, name_file,**parameters):
    name = instance_name(name_file)
    # print("name: ", name[count])
    name[count] = name[count].replace('\t', '')
    # print("parameters:" , parameters['parameters'])
    # print("DATA FOUND!!!", 'inputs/instances/Test/' + name[count] +'_')

    # try:
    data = pd.read_csv('inputs/instances/Test/' + name[count], sep='\t', header=None, skiprows=[0, 1])
    data1 = pd.read_csv('inputs/instances/Test/' + name[count], header=None)
    # print("DATA FOUND!!!", 'inputs/instances/Test/' + name[count])
    # except:
    #     print("NOT FOUND!!!", 'inputs/instances/Test/' + name[count])
    #     return "not found"

    roll_count = int(data1.iloc[0])
    roll_length = int(data1.iloc[1])
    order_length = data[0].tolist()
    order_count = data[1].tolist()
    name_ = name[count]
    # print(roll_count)
    INSTANCE = CuttingStock(order_count, order_length, roll_length, name_, parameters=parameters['parameters'])
    return INSTANCE


def instance_val(count, name_file, **parameters):
    name = instance_name(name_file)
    try:
        # data = pd.read_csv('/content/drive/MyDrive/MIE1666/Test/'+name[count],delimiter = "\t",error_bad_lines=False, header = None,skiprows=[0,1])
        # data1 = pd.read_csv('/content/drive/MyDrive/MIE1666/Test/'+name[count],header = None)

        data = pd.read_csv('instances/Validation/' + name[count], delimiter="\t", header=None, skiprows=[0, 1])
        data1 = pd.read_csv('instances/Validation/' + name[count], header=None)
    except:
        return "not found"

    roll_count = int(data1.iloc[0])
    roll_length = int(data1.iloc[1])
    order_length = data[0].tolist()
    order_count = data[1].tolist()
    name_ = name[count]
    # print(roll_count)
    INSTANCE = CuttingStock(order_count, order_length, roll_length, name_)
    return INSTANCE
