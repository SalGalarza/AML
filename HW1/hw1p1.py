### Applied Machine Learning - Homework 1
## Salvador Galarza

from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
from PIL import Image
import scipy.spatial
import random
from sklearn.neighbors import KNeighborsClassifier
import pickle
import scikitplot as skplt
from sklearn import metrics
from sklearn import preprocessing
import json 

## Join the Digit Recognizer competition on Kaggle. Download the training and test data. The
## competition page describes how these files are formatted.

#Reading data file
train_path = "hw1/train.csv"
test_path = "hw1/test.csv"

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
def display_unique_digits(data):

    # unique_df = df_train.drop_duplicates(subset = ["label"])
    # unique_np = unique_df.to_numpy()
    unique_np = data
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

## (d) Pick one example of each digit from your training data. Then, for each sample digit, compute
# and show the best match (nearest neighbor) between your chosen sample and the rest of
# the training data. Use L2 distance between the two images’ pixel values as the metric.

def l2_distance():
    unique = get_unique_df()
    distances = []
    s = [1,7,3,9,2,6,0,4,8,5]
    cc = dict.fromkeys(s,[])

    for i in range(len(unique)):
        l = []
        for ii in range(len(np_train)):
            dist = scipy.spatial.distance.cdist([unique[i][1:]],[np_train[ii][1:]])
            l.append(dist)
            cc[s[i]] = l
            # distances.append(dist)
    for d in cc:
        cc[d].remove([[[0.0]]])
        minval =  min(cc[d])[0][0]
        mindex = cc[d].index(minval)
        minlable = np_train[mindex][0]
        cc[d] = [minlable,mindex,minval]
    
    # with open("sample.json", "w") as outfile:
    #     json.dump(cc, outfile)
        

    return cc

# x = l2_distance()
# print(x)
# print(min(x['0-0']))
# print(min(x['1-1']))
# print(min(x['0-1']))

## (e) Consider the case of binary comparison between the digits 0 and 1. Ignoring all the other
# digits, compute the pairwise distances for all genuine matches and all impostor matches,
# again using the L2 norm. Plot histograms of the genuine and impostor distances on the same
# set of axes.

def binary_comp(a2):
    # unique_np = a1
    df_train = a2
    # zero_np = unique_np[1]
    # one_np = unique_np[0]
    zeros_train_df = df_train[(df_train["label"]==0)] 
    ones_train_df = df_train[(df_train["label"]==1)] 
    zeros_train_np = zeros_train_df.to_numpy()
    ones_train_np = ones_train_df.to_numpy()

    zero_zero_dist = scipy.spatial.distance.cdist(zeros_train_np,zeros_train_np)
    one_one_dist = scipy.spatial.distance.cdist(ones_train_np,ones_train_np)
    zero_one_dist = scipy.spatial.distance.cdist(zeros_train_np,ones_train_np)

    dist_dict = {"0-0": zero_zero_dist, "1-1": one_one_dist, "0-1": zero_one_dist}

    # dist_dict = array1
    zz = dist_dict["0-0"]
    oo = dist_dict["0-1"]
    zo = dist_dict["1-1"]

    # plot_binary_comp(dist_dict)

    zznp = zz[np.triu_indices(4132)]
    oonp = oo[np.triu_indices(4132)]
    zonp = zo[np.triu_indices(4132)]

    dist_dict["0-0"] = zznp[zznp != 0]
    dist_dict["0-1"] = oonp[oonp != 0]
    dist_dict["1-1"] = zonp[zonp != 0]

    # plot_binary_comp(dist_dict)

    return dist_dict

def plot_binary_comp(dist_dict):

    fig = plt.figure(1)
    plt.hist(dist_dict["0-0"],color='C0',edgecolor='white', rwidth=1)
    plt.title('Genuine Matches (0-0)')
    plt.xlabel('Pairs')
    plt.ylabel('Distance')
    plt.savefig("0-0-Hist.png",bbox_inches='tight')

    fig = plt.figure(2)
    plt.hist(dist_dict["0-1"],color='C1',edgecolor='white', rwidth=1)
    plt.title('Impostor Matches (0-1)')
    plt.xlabel('Pairs')
    plt.ylabel('Distance')
    plt.savefig("0-1-Hist.png",bbox_inches='tight')

    fig = plt.figure(3)
    plt.hist(dist_dict["1-1"],color='C2',edgecolor='white', rwidth=1)
    plt.title('Genuine Matches (1-1)')
    plt.xlabel('Pairs')
    plt.ylabel('Distance')
    plt.savefig("1-1-Hist.png",bbox_inches='tight')

    # #combo graph
    fig = plt.figure(4)
    b, bins, patches = plt.hist([dist_dict["0-0"],dist_dict["0-1"],dist_dict["1-1"]],label=['(0-0)', '(0-1)', '(1-1)'])
    plt.legend()
    plt.title('All Matches (0-0),(0-1),(1-1)')
    plt.xlabel('Pairs')
    plt.ylabel('Distance')
    plt.savefig("All-Hist.png",bbox_inches='tight')

    return None


# (f ) Generate an ROC curve from the above sets of distances. What is the equal error rate? What
# is the error rate of a classifier that simply guesses randomly?

def roc():
    dist_dict = binary_comp(df_train)

    y_true = [1]*len(dist_dict['0-0']) + [1]*len(dist_dict['1-1']) + [0]*len(dist_dict['0-1'])
    # 8534646, 8534646, 8538778

    yy = np.concatenate((dist_dict['0-0'], dist_dict['1-1']), axis=None)
    y_probas = np.concatenate((yy, dist_dict['0-1']), axis=None)
    norm = np.linalg.norm(y_probas)
    normal_array = y_probas/norm

    # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas, pos_label=2)
    # y_probas = dist_dict['0-0'] + dist_dict['1-1'] + dist_dict['1-0']
    # predicted probabilities generated by sklearn classifier
    skplt.metrics.plot_roc(y_true, normal_array)
    # plt.plot(y_true,normal_array)
    plt.savefig("roc2.png",bbox_inches='tight')
    plt.show()

    # return len(y_true),len(y_probas),len(dist_dict['0-0']),len(dist_dict['1-1']),len(dist_dict['0-1'])
    # return fpr, tpr, thresholds
    return "done"

# (g) Implement a K-NN classifier. (You cannot use external libraries that implement K-NN for this
# question; it should be your own implementation. You can still use libraries such as numpy,
# scipy, pandas, etc., for data manipulation.)

def knn(point,k,data):

    distances = scipy.spatial.distance.cdist([point],data)
    ordered_vals = np.argpartition(distances[0],k)
    # k_min_vals = distances[0][ordered_vals[:k]]
    s = ordered_vals[1:k+1]
    l=[]
    for i in s:
        l.append(data[i][0])

    # k_min_vals_dict = dict.fromkeys(s,0)
    # for i in range(k):
    #     k_min_vals_dict[s[i]] = k_min_vals[i]

    return max(set(l), key=l.count)

# print(knn(np_train[0],3,np_train))

## (h) Randomly split the training data into two halves. Train your k-NN classifier on the first half
# of the data, and test it on the second half, reporting your average accuracy. Note: If you find
# your implementation to be slow, you can instead select a subsample of 5,000 from the test set
# rather than the entire test set.

def split_data(data):
    data = data.sample(frac=1)
    np_train = df_train.to_numpy()
    i = int(len(np_train)/2)
    train_data = np_train[:i]
    test_data = np_train[i:]

    return train_data,test_data

def test_knn(data):
    k = 3
    s_data = split_data(data)
    train_data = s_data[0]
    test_data = s_data[1][:5000]
    predictions = []
    for dt_pnt in test_data:
        prediction = knn(dt_pnt,k,train_data)
        predictions.append((dt_pnt[0], prediction))
    
    return predictions

def knn_pred(a1,a2):
    knn = KNeighborsClassifier(n_neighbors=5)

    df_train = a1.drop('label', 1)

    #Train the model using the training sets
    knn_model = knn.fit(df_train, a1['label'])

    filename = 'knn_model.sav'
    pickle.dump(knn_model, open(filename, 'wb'))

    #Predict the response for test dataset
    y_pred = knn.predict(a2)

    # np.savetxt("predictions.csv", y_pred, delimiter = " ")

    df = pd.DataFrame(y_pred, columns=["predictions"])
    df.to_csv('predictions.csv', index=False)

    return "done"

def knn_prediction(a1):
    #Predict the response for test dataset
    # y_pred = knn.predict(a1)

    knn_pred = pickle.load(open('hw1/p2_titanic/knn_model.sav', 'rb'))
    y_pred = knn_pred.predict(a1)

    # np.savetxt("predictions.csv", y_pred, delimiter = " ")

    df = pd.DataFrame(y_pred, columns=["predictions"])
    df.to_csv('predictions.csv', index=False)

    return "done"

    


# print(roc())
# digit_count_hist()

# df_train_label = df_train.drop('label', 1)

# print(df_train_label.shape)
# print(df_test.shape)
# print(knn_pred(df_train,df_test))

# print(knn_prediction(df_test))



# plot_binary_comp(binary_comp(df_train))
# display_unique_digits(unique_data)





# print(l2_distance())




# def l2_distance(mydigit,otherdigit):
#     l = []
#     for i in range(len(mydigit)):
#         l.append((mydigit[i]-otherdigit[i])**2)

#     return sum(l)**.5

# final_distances = []
# unique_digit_data = get_unique_df()
# for my_digit_data in unique_digit_data:
#     distances = []
#     for digit_data in np_train:
#         distances.append(l2_distance(my_digit_data,digit_data))
#     # remove if min dist == 0 remove min dist and get new min dist
#     # we also need to get label of such min distance
#     final_distances.append(min(distances))
# print(final_distances)



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

