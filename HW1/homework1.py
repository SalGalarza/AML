### Applied Machine Learning - Homework 1
## Salvador Galarza

from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
from PIL import Image


## Join the Digit Recognizer competition on Kaggle. Download the training and test data. The
## competition page describes how these files are formatted.

#Reading data file
train_path = "hw1/train.csv"
test_path = "hw1/train.csv"

# file = open("hw1/train.csv")
# data = csv.DictReader(file)
# df_test = pd.read_csv(test_path, index_col="label")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

np_train = df_train.to_numpy()

def get_unique_df():
    unique_df = df_train.drop_duplicates(subset = ["label"])
    unique_np = unique_df.to_numpy()
    return unique_np

## (b) function to display an MNIST digit. Display one of each digit.
def display_unique_digits():

    unique_df = df_train.drop_duplicates(subset = ["label"])
    unique_np = unique_df.to_numpy()
    # for i in range(len(unique_np)):
    #     # print(unique_np[i][1:])
    #     labels = unique_np[i][0]
    #     num_vals = unique_np[i][1:]
    #     numarray = num_vals.reshape((28, 28)).astype('uint8')
    #     img = Image.fromarray(numarray)
    #     img.save('image{}.png'.format(labels))

    for i in range(len(unique_np)):
        # print(unique_np[i][1:])
        labels = unique_np[i][0]
        num_vals = unique_np[i][1:]
        numarray = num_vals.reshape(28, 28)

        fig = plt.figure(i)
        plt.xticks([])
        plt.yticks([])
        plt.title("label: {}".format(labels))
        plt.imshow(numarray)
        plt.gray()
        plt.savefig('image_{}.png'.format(labels))


## (c) Examine the prior probability of the classes in the training data. Is it uniform across the 
## digits? Display a normalized histogram of digit counts. Is it even?

def digit_count_hist():
    digit_count = df_train['label'].value_counts()
    total = sum(digit_count)
    prob = digit_count/total
    print(prob)
    ## It is not entierly uniform but they are fairly close.

    norm_digit_count = (digit_count-min(digit_count))/(max(digit_count)-min(digit_count))
    # x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
    # num_bins = len(digit_count)
    names = [1,7,3,9,2,6,0,4,8,5]
    # plt.axis([1,9,0,1])
    plt.bar(names,norm_digit_count, width=1, edgecolor='black')
    plt.xticks(names)
    # plt.yticks(x)
    plt.xlabel('Digits')
    plt.ylabel('Digit Count')
    plt.savefig("DigitCount_Hist.png")

    ## No it is not even.

def l2_distance(mydigit,otherdigit):
    l = []
    for i in range(len(mydigit)):
        l.append((mydigit[i]-otherdigit[i])**2)

    return sum(l)**.5

final_distances = []
unique_digit_data = get_unique_df()
for my_digit_data in unique_digit_data:
    distances = []
    for digit_data in np_train:
        distances.append(l2_distance(my_digit_data,digit_data))
    # remove if min dist == 0 remove min dist and get new min dist
    # we also need to get label of such min distance
    final_distances.append(min(distances))
print(final_distances)



# plt.show()

    # print(num_d[0])

# digit_count_hist()

# digit_count_hist()
# np_df = df_test.to_numpy()
# print(np_df['0'].value_counts())





# # main function
# def main():
#     # display_unique_digits()


# main()

# num_vals = unique_np[0][1:]
# numarray = num_vals.reshape(28, 28)

# fig = plt.figure(1)
# plt.imshow(numarray)
# plt.gray()
# plt.savefig("IrisData_Plot.png")

# print( unique_np[0][1:] )


# print(df_train['label'][0])

# fig = plt.figure
# img = unique_np[0][1:]
# plt.imshow(img)

#Extracting data from each column based on column name
# column['label']

#Creating numpy array/matrix of the attributes
# nparray = np.array([sepal_length, sepal_width, petal_length, petal_width],dtype=object)


# Code Reference 

