import numpy as np
import matplotlib.pyplot as plt

def generateDDR(D):
    randomHV=np.ones(D)
    permuIdx=np.random.permutation(D)
    randomHV[permuIdx[:int(D/2)]] = -1
    # for i in range(int(len(permuIdx)/2)):
    #     randomHV[permuIdx[i]]=-1
    randomHV=randomHV.astype(int)
    return randomHV

def binarizeMajorityRule(HV):
    hv = np.ones(len(HV), dtype='float32')
    for i in range(len(HV)):
        if HV[i] < 0:
            hv[i] = -1
    return hv

def encoding(sampleX,D,pixelMemory,valueMemory):
    sampleHV=np.zeros(D, dtype='float32')
    num_pixels=len(sampleX)
    for i in range(num_pixels):
        sampleHV += pixelMemory[i]*valueMemory[sampleX[i]]
    # return sampleHV                             # Non-binary
    return binarizeMajorityRule(sampleHV)     # Binary
    
def inference(y_test, test_HVs, associativeMemory, num_class):
    num_class = num_class
    length = len(test_HVs)
    all_list = np.zeros(num_class)
    correct_list = np.zeros(num_class)

    for i in range(length):
        hamm_list = []
        Y = test_HVs[i]
        inds = []
        for j in range(num_class):
            item  = np.dot(Y, associativeMemory[j])
            hamm_list.append(item)
        pred_class = np.argmax(hamm_list)
        all_list[int(y_test[i])] = all_list[int(y_test[i])] + 1
        if pred_class == y_test[i]:
            correct_list[int(y_test[i])] = correct_list[int(y_test[i])] + 1

    sum_corr = 0
    sum_all = len(y_test)
    for i in range(num_class):
        sum_corr += correct_list[i]
    print('Avg: %.4f'%(sum_corr/sum_all))
    return sum_corr/sum_all
