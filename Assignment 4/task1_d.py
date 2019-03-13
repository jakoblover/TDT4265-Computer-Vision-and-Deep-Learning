precision_1 =  [1.0, 1.0, 1.0, 0.5, 0.20]
recall_1 = [0.05, 0.1, 0.4, 0.7, 1.0]
precision_2 = [1.0, 0.80, 0.60, 0.5, 0.20]
recall_2 = [0.3, 0.4, 0, 5, 0.7, 1.0]

avg_precision_1 = 0
avg_precision_2 = 0
for recall_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #Get indices where recall value is greater than recall_level
    idx_1 = [i for i,j in enumerate(recall_1) if j >= recall_level]
    idx_2 = [i for i, j in enumerate(recall_2) if j >= recall_level]
    #Get list of precision values using these indices
    precision_list_1 = [precision_1[i] for i in idx_1]
    precision_list_2 = [precision_2[i] for i in idx_1]
    #Add the maximum value....
    avg_precision_1 += max(precision_list_1)
    avg_precision_2 += max(precision_list_2)
#...then take the average
avg_precision_1 = avg_precision_1 / 11
avg_precision_2 = avg_precision_2 / 11

print("Average precision of class 1: ", avg_precision_1)
print("Average precision of class 2: ", avg_precision_2)
print("Mean average precision: ", (avg_precision_1+avg_precision_2)/2)

