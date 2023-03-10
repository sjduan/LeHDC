import numpy as np
import matplotlib.pyplot as plt

#################################################
# This is the dataset configuration:
num_feature = 784
num_value = 256
num_class = 10
dimension = 10000       # this can be changed for the hypervector dimension


#################################################
# Fashion-MNIST
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Use standard FashionMNIST dataset
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True	# please turn it off once downloaded
)

# Use standard FashionMNIST dataset
test_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = True	# please turn it off once downloaded
)

x_train = train_set.train_data.numpy()
x_test = test_set.test_data.numpy()
y_train = train_set.train_labels.numpy()
y_test = test_set.test_labels.numpy()


def reshapeX(x):
    x_reshape=[]
    num=x.shape[0]
    for i in range(num):
        x_reshape.append(x[i].flatten())
    return x_reshape

x_train=reshapeX(x_train)
x_test=reshapeX(x_test)
x_train = np.array(x_train,dtype=int)
y_train = np.array(y_train,dtype=int)
x_test = np.array(x_test,dtype=int)
y_test = np.array(y_test,dtype=int)
print('Dataset MNIST loaded.')

###############Binary HDC Encoding##############################
###################(START)#####################################
if 'int' not in x_train.dtype.name or 'int' not in x_test.dtype.name:
    raise TypeError('the sample datatype is not correct. "int" is needed.')
    
from HDC import *

print('Generating feature and value hypervectors....')
featureMemory = np.zeros((num_feature, dimension))      # generate featureMemory
valueMemory = np.zeros((num_value, dimension))          # generate valueMemory
k = 0
while k < num_feature:
    item = generateDDR(dimension)
    if True or item.tolist() not in featureMemory.tolist():
        featureMemory[k] = item
        k += 1
    del item

item = generateDDR(dimension)
step = dimension / num_value
k = 0
while k < num_value:
    valueMemory[k, :] = item
    valueMemory[k, :np.round(step*k).astype(int)] *= -1
    k += 1

print('HV generation completed.')

print('start prepare HVs for training samples....')
train_HVs = []                                                              # encode the training samples
for i in range(len(x_train)):
    train_HVs.append(encoding(x_train[i], dimension, featureMemory, valueMemory))
print('HVs for training samples are completed.')

print('start prepare HVs for testing samples....')
test_HVs = []                                                               # encode the testing samples
for i in range(len(x_test)):
    test_HVs.append(encoding(x_test[i], dimension, featureMemory, valueMemory))
print('HVs for testing samples are completed.')

train_HVs = np.array(train_HVs)
test_HVs = np.array(test_HVs)

ClassHVs = np.zeros((num_class, dimension))
for i in range(x_train.shape[0]):
    ClassHVs[y_train[i]] += train_HVs[i]

associativeMemory = []                                                # get the class hypervectors, called "associativeMemory"
for i in range(num_class):
    associativeMemory.append(binarizeMajorityRule(ClassHVs[i]))
associativeMemory = np.array(associativeMemory)

# Tips: the varaibles ['train_HVs','test_HVs','associativeMemory'] can be stored for later use, and you don't need to run this Encoding block anymore.
###################(END)########################################



######################### HDC Inference ##########################
# Tips: This block can be run anytime after you generate the 'associativeMemory', to check the inference accuracy. 
print('Binary HDC Inference:')
inference(y_test, test_HVs, associativeMemory, num_class)
##################################################################



############################## BNN Training ########################
from BNN import *
print('Start BNN training...')

torch.seed()
x_train = torch.from_numpy(train_HVs.reshape((x_train.shape[0],-1))).type(torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.int64).flatten().to(device)
train_set = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)       # The batch_size can be fine-tuned.
x_test = torch.from_numpy(test_HVs.reshape((len(x_test),-1))).type(torch.float).to(device)
y_test = torch.tensor(y_test, dtype=torch.int64).flatten().to(device)
test_set = torch.utils.data.TensorDataset(x_test, y_test)

D = dimension
num_class = num_class
dropout_rate = 0.5
weight_decay = 5e-2
learning_rate = 1e-2

model = BHDC(inshape=D, outshape=num_class, dropout_prob=dropout_rate).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr = learning_rate)

### this is the training step
epochs = 100
loss_accum = 0
loss_log = 0

train_BNN_acc = []
test_BNN_acc = []

for i in range(epochs):
    model.train()
    for _, batch in enumerate(trainloader):
        batch_train, batch_label = batch
        y_pred = model(batch_train)
        loss = criterion(y_pred, batch_label.flatten().to(device))
        if np.random.rand() < 0.2:
            loss_accum += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i%1 == 0 or i == epochs - 1:
        model.eval()
        print(i, loss_accum, end='\t')
        with torch.no_grad():
            outputs = model(x_test)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == y_test).sum().item()/len(y_test)
            print(acc)
            test_BNN_acc.append(acc)

            outputs = model(x_train)
            _, predicted = torch.max(outputs.data, 1)
            tacc = (predicted == y_train).sum().item()/len(y_train)
            train_BNN_acc.append(tacc)

        if loss_accum > loss_log:   # Adaptive learning rate, once the training loss decrease, I make the lr be 0.5 times.
            optimizer.param_groups[0]['lr'] *= 0.5
        loss_log = loss_accum
        loss_accum = 0
        
#### To extract the weights as the "associativeMemory"
print('Extracting weights as the class hypervectors...')
real_weights = model.binary_weight_l2.weight
binary_weights = torch.sign(real_weights)
associativeMemory = binary_weights.cpu().detach().numpy()

#### now we can run the inference again to test the accuracy
print('Binary HDC Inference:')
inference(y_test, test_HVs, associativeMemory, num_class)
