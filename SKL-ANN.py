from pylab import *
from csv import reader
from sklearn.neural_network import MLPClassifier




# LOAD Training Data
filename = 'isolet1+2+3+4.data'
dataset = list()
with open(filename, 'r') as file:
	csv_reader = reader(file)
	for row in csv_reader:
		if not row:
			continue
		dataset.append(row)

# Separate features(X) from Class(Y)
for i in range(0, len(dataset[0])):
	for row in dataset:
		row[i] = float(row[i].strip())

X_Tr = [row[0:-1] for row in dataset]
X_Tr = np.array(X_Tr)

Y_Tr = [row[-1] for row in dataset]
Y_Tr = np.array(Y_Tr)



# LOAD Test Data
filename = 'isolet5.data'
dataset = list()
with open(filename, 'r') as file:
	csv_reader = reader(file)
	for row in csv_reader:
		if not row:
			continue
		dataset.append(row)

# Separate features(X) from Class(Y)
for i in range(0, len(dataset[0])):
	for row in dataset:
		row[i] = float(row[i].strip())

X_Ts = [row[0:-1] for row in dataset]
X_Ts = np.array(X_Ts)

Y_Ts = [row[-1] for row in dataset]
Y_Ts = np.array(Y_Ts)

OLSize = 26
ILSize = X_Tr.shape[1]
HLSize = int(round(sqrt(OLSize*ILSize)))

MLC = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(HLSize), random_state=1)

MLC.fit(X_Tr,Y_Tr)

Pr_Tr = MLC.predict(X_Tr)
err_Tr = len(find(Pr_Tr != Y_Tr))/float(len(Y_Tr))
print('Error on Training Set', err_Tr)

Pr_Ts = MLC.predict(X_Ts)
err_Ts = len(find(Pr_Ts != Y_Ts))/float(len(Y_Ts))
print('Error on Test Set', err_Ts)






