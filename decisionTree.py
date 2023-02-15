import numpy as np
from math import log2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy.interpolate import lagrange

def main():
    # data  = readFile('./data/D1.txt')

    # root = buildDecisionTree(data)
    # printTree([root])
    #q2()
    #q3()
    #q4()
    # q5()
    # q6()
    # q7()
    # section3sklearn()
    #section4lagrange()
    pass

class Node:
    def __init__(self, split=None, label=None):
        self.split = split
        self.label = label
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self):
        self.root = None
    
    #define setters

    def setRoot(self, root):
        self.root = root

    #define getters
    def getRoot(self):
        return self.root


def readFile(filename):
    return np.loadtxt(filename)


def determineCandidateSplits(data):
    splits = []
    sortedData = []
    for i in range(2):
        sorted = data[data[:, i].argsort()]
        for j in range(len(sorted)-1):
            if sorted[j][2] == sorted[j+1][2]:
                continue
            else:
                splits.append((sorted[j][i], i))

    return splits


def findBestSplit(data):
    
    splits = determineCandidateSplits(data)
    infoGains = []

    numClass1 = list(data[:, 2]).count(0)
    numClass2 = list(data[:, 2]).count(1)

    if(numClass1 == 0 or numClass2 == 0):
        entropyY = 0
    else:
        entropyY = -(numClass1/(numClass1+numClass2))*log2(numClass1/(numClass1+numClass2))
        entropyY += -(numClass2/(numClass1+numClass2))*log2(numClass2/(numClass1+numClass2))

    gainRatio = -1
    bestSplit = -1
    bestLeft = np.array([])
    bestRight = np.array([])

    for split in splits:
        left, right = splitData(data, split)


        if(left.shape[0] != 0):
            class1CountRatioLeft = list(left[:, 2]).count(0)/ left.shape[0]
            class2CountRatioLeft = list(left[:, 2]).count(1)/ left.shape[0]
        else:
            class1CountRatioLeft = 0
            class2CountRatioLeft = 0

        if(right.shape[0]!= 0):
            class1CountRatioRight = list(right[:, 2]).count(0)/ right.shape[0]
            class2CountRatioRight = list(right[:, 2]).count(1)/ right.shape[0]
        else:
            class1CountRatioRight = 0
            class2CountRatioRight = 0

        if(class1CountRatioLeft == 0):
            leftEntropy = 0
        else:
            leftEntropy = -(class1CountRatioLeft*log2(class1CountRatioLeft))
        if(class2CountRatioLeft == 0):
            leftEntropy += 0
        else:
            leftEntropy += -(class2CountRatioLeft*log2(class2CountRatioLeft))
        if(class1CountRatioRight == 0):
            rightEntropy = 0
        else:
            rightEntropy = -(class1CountRatioRight*log2(class1CountRatioRight))
        if(class2CountRatioRight == 0):
            rightEntropy += 0
        else:
            rightEntropy += -(class2CountRatioRight*log2(class2CountRatioRight))
        

        infoGain = entropyY 
        infoGain -= ((left.shape[0]/data.shape[0])*leftEntropy)  
        infoGain -= ((right.shape[0]/data.shape[0])*rightEntropy)

        splitRatio = left.shape[0]/data.shape[0]
        if(splitRatio == 0 or splitRatio == 1):
            infoGains.append(infoGain)
            continue
        splitEntropy = -(splitRatio*log2(splitRatio))
        splitEntropy -= (1-splitRatio)*log2(1-splitRatio)

        if(splitEntropy == 0):
            infoGains.append(infoGain)
            continue

        newGainRatio = infoGain/splitEntropy
        infoGains.append(newGainRatio)
        if gainRatio < newGainRatio:
            gainRatio = newGainRatio
            bestSplit = split
            bestLeft = left
            bestRight = right

        if(gainRatio == 0):
            continue

    #for i in range(len(infoGains)):
    #    print(str(splits[i]) +'  InfoGainRatio: ' + str(infoGains[i]))
    #print('--------------------------------')
    return bestSplit, gainRatio

def splitData(data, split):
    left = data[data[:,split[1]] >= split[0]]
    right = data[data[:,split[1]] < split[0]]
    return left, right

def buildDecisionTree(data):
    (bestSplit, bestInfoGainRatio) = findBestSplit(data)
    #if stopping criterion is met, bestSplit is -1
    if bestSplit == -1:
        leaf = Node(None, 1)
        if list(data[:, 2]).count(0) > list(data[:, 2]).count(1):
            leaf.label = 0
        return leaf

    #while stopping crtieria not met

    root = Node(bestSplit)
    left, right = splitData(data, bestSplit)
    root.left = buildDecisionTree(left)
    root.right = buildDecisionTree(right)


    return root



def printTree(nodes):
    if len(nodes) == 0:
        return
    newLevelNodes = []
    for root in nodes:
        if root.left is not None:
            newLevelNodes.append(root.left)
        if root.right is not None:
            newLevelNodes.append(root.right)
    for n in nodes:
        if n.split is not None:
            print(n.split, end=" ")
        if n.label is not None:
            print(n.label, end=" ")
    print(('\n-----------------------------------------------------------------------'
    '----------------------------------------------------------------------------------'))
     
    printTree(newLevelNodes)
    

def classifyPoint(root, point):
    if root.split is None:
        return root.label
    else:
        dim = root.split[1]
        threshold = root.split[0]
        if(point[dim] >= threshold):
            return classifyPoint(root.left, point)
        else:
            return classifyPoint(root.right, point)





def q2():
    p1 = [1,1]
    p2 = [1,2]

    p3 = [2,1]
    p4 = [2,2]

    data1 = [p1, p4]
    data2 = [p2, p3]

    plt.scatter([1,2],[1,2], s=200)
    plt.scatter([1,2], [2,1],s=200)

    plt.show()

def q3():
    data = readFile('./data/Druns.txt')
    buildDecisionTree(data)

def q4():
    data = readFile('./data/D3leaves.txt')
    root = buildDecisionTree(data)
    printTree([root])

def q5():
    data = readFile('./data/D2.txt')
    root = buildDecisionTree(data)
    printTree([root])

def q6():
    data = readFile('./data/D1.txt')
    root1 = buildDecisionTree(data)
    points1 = data[:, 0:2]
    labels = []

    for p in points1:
        labels.append(classifyPoint(root1, p))
    
    plt.scatter(points1[:, 0], points1[:, 1], c=labels)
    plt.show()


    data = readFile('./data/D2.txt')
    root2 = buildDecisionTree(data)

    points2 = data[:, 0:2]
    labels = []

    for p in points2:
        labels.append(classifyPoint(root2, p))
    
    plt.scatter(points2[:, 0], points2[:, 1], c=labels)

    plt.show()

def countNodes(root):
    if root is None:
        return 0
    else:
        return 1 + countNodes(root.left) + countNodes(root.right)


def q7():
    data = readFile('./data/Dbig.txt')
    train, test = train_test_split(data, train_size = 8192)

    D1 = train[:32, :]
    D2 = train[:128, :]
    D3 = train[:512, :]
    D4 = train[:2048, :]
    D5 = train

    sets = [D1, D2, D3, D4, D5]
    testErrors = []
    numNodes = []
    for D in sets:
        root = buildDecisionTree(D)
        points = D[:, 0:2]
        labels = []

        for p in test[:, 0:2]:
            labels.append(classifyPoint(root, p))

        err = np.count_nonzero(labels - test[:, 2])
        testErrors.append(err)

        numNodes.append(countNodes(root))

        #plt.scatter(test[:, 0], test[:, 1], c=labels)
        #plt.show()

    # plt.plot(numNodes, testErrors)
    # plt.show()
    print(numNodes)
    print(testErrors)

    
def section3sklearn():
    data = readFile('./data/Dbig.txt')
    train, test = train_test_split(data, train_size = 8192)

    D1 = train[:32, :]
    D2 = train[:128, :]
    D3 = train[:512, :]
    D4 = train[:2048, :]
    D5 = train

    sets = [D1, D2, D3, D4, D5]

    testErrors = []
    numNodes = []
    for D in sets:
        tree = DecisionTreeClassifier()
        tree.fit(D[:,0:2], D[:, 2])

        labels = tree.predict(test[:, 0:2])

        err = np.count_nonzero(labels - test[:, 2])
        testErrors.append(err)

        numNodes.append(tree.tree_.node_count)

    print(numNodes)
    print(testErrors)
    plt.plot(numNodes, testErrors)
    plt.show()
    





def section4lagrange():
    x = np.random.uniform(0,100,100)
    y = np.sin(x)

    x_test = np.random.uniform(0,1,100)

    poly = lagrange(x,y)
    y_pred_test = poly(x_test)
    y_pred_train = poly(x)

    train_error = mean_squared_error(y, y_pred_train)
    test_error = mean_squared_error(y, y_pred_test)

    noise1 = np.random.normal(0, 100, 100)
    xWithNoise1 = x + noise1

    noise2 = np.random.normal(0, 200, 100)
    xWithNoise2 = x + noise2

    y_pred_test2 = poly(xWithNoise1)
    y_pred_train2 = poly(xWithNoise1)

    train_error2 = mean_squared_error(y, y_pred_train)
    test_error2 = mean_squared_error(y, y_pred_test)

    y_pred_test3 = poly(xWithNoise2)
    y_pred_train3 = poly(xWithNoise2)

    train_error3 = mean_squared_error(y, y_pred_train)
    test_error3 = mean_squared_error(y, y_pred_test)

    print('No Noise, Training Error Rate: ' + str(train_error))
    print('No Noise, Test Error Rate: ' + str(test_error))
    print('Gaussian Noise with Var=1, Training Error Rate: ' + str(train_error2))
    print('Gaussian Noise with Var=1, Test Error Rate: ' + str(test_error2))
    print('Gaussian Noise with Var=2, Training Error Rate: ' + str(train_error3))
    print('Gaussian Noise with Var=2, Test Error Rate: ' + str(test_error3))


if __name__ == '__main__':
    main()
