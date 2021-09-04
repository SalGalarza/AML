### Applied Machine Learning - Homework 0
## Salvador Galarza

from matplotlib import pyplot as plt
import numpy as np
import csv

#Reading data file
file = open("/Users/salgalarza/Documents/AML/hw0/iris.data.csv")
data = csv.DictReader(file)

# Creating lists that will store each column data
sepal_length = []
sepal_width = []
petal_length = []
petal_width = []
label = []

#Extracting data from each column based on column name
for column in data:
    sepal_length.append(column['sepal_length'])
    sepal_width.append(column['sepal_width'])
    petal_length.append(column['petal_length'])
    petal_width.append(column['petal_width'])
    label.append(column['label'])

#Creating numpy array/matrix of the attributes
nparray = np.array([sepal_length, sepal_width, petal_length, petal_width],dtype=object)

#Setting the colors for each of the 50 data points per sample
colors = ["r"]*50+["g"]*50+["b"]*50

#Creating plot figure and subplots
fig =plt.figure(0)

a11 = fig.add_subplot(441)

a1 = fig.add_subplot(442)
a2 = fig.add_subplot(443)
a3 = fig.add_subplot(444)

b22 = fig.add_subplot(446)

b1 = fig.add_subplot(445)
b2 = fig.add_subplot(447)
b3 = fig.add_subplot(448)

c33 = fig.add_subplot(4,4,11)

c1 = fig.add_subplot(4,4,9)
c2 = fig.add_subplot(4,4,10)
c3 = fig.add_subplot(4,4,12)

d44 = fig.add_subplot(4,4,16)

d1 = fig.add_subplot(4,4,13)
d2 = fig.add_subplot(4,4,14)
d3 = fig.add_subplot(4,4,15)

#Creating Scatter Plots
a11.text(.5,.5,"Sepal.Length", size=15, ha='center', va='center')

a1.scatter([nparray[1]], [nparray[0]], c=colors)

a2.scatter([nparray[2]], [nparray[0]], c=colors)

a3.scatter([nparray[3]], [nparray[0]], c=colors)

b22.text(0.5,0.5,"Sepal.Width", size=15, ha='center', va='center')

b1.scatter([nparray[0]], [nparray[1]], c=colors)

b2.scatter([nparray[2]], [nparray[1]], c=colors)

b3.scatter([nparray[3]], [nparray[1]], c=colors)

c33.text(.5,.5,"Petal.Length", size=15, ha='center', va='center')

c1.scatter([nparray[0]], [nparray[2]], c=colors)

c2.scatter([nparray[1]], [nparray[2]], c=colors)

c3.scatter([nparray[3]], [nparray[2]], c=colors)

d44.text(.5,.5,"Petal.Width", size=15, ha='center', va='center')

d1.scatter([nparray[0]], [nparray[3]], c=colors)

d2.scatter([nparray[1]], [nparray[3]], c=colors)

d3.scatter([nparray[2]], [nparray[3]], c=colors)

fig.suptitle("Iris Data (red=setosa,green=versicolor,blue=virginica)",fontweight='bold')

#Adjusting Axis
plts = [a11,a1,a2,a3,b22,b1,b2,b3,c33,c1,c2,c3,d44,d1,d2,d3]
for subplt in plts:
    subplt.set_xticks([])
    subplt.set_yticks([])

axis1 = [1,2,3,4,5,6,7]
axis2 = [2,2.5,3,3.5,4]
axis3 = [.5,1,1.5,2,2.5]
axis4 = [4.5,5.5,6.5,7.5]

a1.xaxis.tick_top()
a1.set_xticks(axis2)

a3.xaxis.tick_top()
a3.set_xticks(axis3)

a3.yaxis.tick_right()
a3.set_yticks(axis4)

b1.set_yticks(axis2)
c3.yaxis.tick_right()
c3.set_yticks(axis1)

d1.set_xticks(axis4)
d1.set_yticks(axis3)
d3.set_xticks(axis1)

#Saving and Showing Final Plot
plt.savefig("IrisData_Plot.png")
plt.show()

# Code Reference 
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
# https://matplotlib.org/stable/tutorials/text/text_intro.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
# https://stackoverflow.com/questions/10354397/python-matplotlib-y-axis-ticks-on-right-side-of-plot