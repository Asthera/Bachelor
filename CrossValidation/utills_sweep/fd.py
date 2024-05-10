import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('general_single.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')

    for row in plots:
        x.append(row[-2])
        y.append(row[1])

plt.bar(x, y, color='g', width=0.72, label="Age")
plt.xlabel('Transforms')
plt.ylabel('Test F1 mean')
plt.title('Single trnasform ')
plt.legend()
plt.show()