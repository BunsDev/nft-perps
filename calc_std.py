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

def read_csv_file(filename):
    #print(filename)

    mark_prices = []
    delta_price = []
    last_time = 0
    last_price = 0
    num_lines = 0

    last_index_price = 0
    delta_index = []

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        #next(reader)  # Skip the header row

        for row in reader:
            curr_time = int(row["block_time"])
            if last_time > 0:
                for t in range(last_time + 1, curr_time):
                    mark_prices.append(last_price)

                delta_price.append(abs(last_price - float(row["mark_price"])))
                delta_index.append(abs(last_index_price - float(row["index_price"])))

            # else
            
            last_time = curr_time

            last_price = float(row["mark_price"])
            last_index_price = float(row["index_price"])
            mark_prices.append(last_price)

            num_lines += 1

    np_arr_price = np.array(mark_prices)
    np_arr_delta = np.array(delta_price)
    print(filename, "std/mean", np.std(np_arr_price) / np.mean(np_arr_price), num_lines)
    print(filename, "mean delta / mean price", np.mean(np_arr_delta) / np.mean(np_arr_price), "std delta / mean price", np.std(np_arr_delta) / np.mean(np_arr_price))
    print(filename, "mean index delta / mean price", np.mean(delta_index) / np.mean(np_arr_price), "std delta / mean price", np.std(delta_index) / np.mean(np_arr_price))

    #return x_values, y_values
'''
# Replace 'data.csv' with the path to your CSV file
files = ["./real_data/azuki_full_trade_data.csv", 
         "./real_data/bayc_full_trade_data.csv",
         "./real_data/mayc_full_trade_data.csv",
         "./real_data/milady_full_trade_data.csv", 
         "./real_data/punks_full_trade_data.csv"]

for filename in files:
	read_csv_file(filename)
	#plot_graph(x_data, y_data, filename)
	
'''