import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def calculate_r2(x, y):
    #print(y)
    mean_y = np.mean(y)
    ss_tot = np.sum((y - mean_y) ** 2)
    ss_res = np.sum((y - x) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def plot_graph(x, y, filename, r2):
    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(os.path.splitext(filename)[0] + " " + "r2 = " + str(r2))
    plt.legend(['mark price', 'index price'])
    png_file = os.path.splitext(filename)[0] + '.png'
    plt.savefig(png_file)
    plt.clf()
    #plt.show()

def read_csv_file(filename):
    print(filename)
    x_values = []
    y_values = []
    

    temp_arr1 = []
    temp_arr2 = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        #next(reader)  # Skip the header row

        for row in reader:
            x_values.append(float(row["block_time"]))
            y_values.append([float(row["mark_price"]), float(row["index_price"])])
            temp_arr1.append(float(row["mark_price"]))
            temp_arr2.append(float(row["index_price"]))

    arr1 = np.array(temp_arr1)
    arr2 = np.array(temp_arr2)
           
    r2 = calculate_r2(arr2, arr1)
    print(r2)
    plot_graph(x_values, y_values, filename, r2)           

    #return x_values, y_values
'''
# Replace 'data.csv' with the path to your CSV file
files = ["with_repeg.csv", "without_repeg.csv"]

for filename in files:
	read_csv_file(filename)
	#plot_graph(x_data, y_data, filename)
	
'''