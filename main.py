import struct
import os
import numpy as np
from annoy import AnnoyIndex

def main():
    print("Getting test data")
    testL, testI = get_test()
    print("Test data complete. Getting training data...")
    trainL, trainI = get_train()    

    #UNCOMMENT TO PRODUCE NEW INDEX

    #print("Training data complete. Indexing...")
    #makeIndex(trainL, trainI)


    print("Index created")
    results = testTrain(testL, testI, trainL, trainI)


def testTrain(testL, testI, trainL, trainI):
    u = AnnoyIndex(784)
    u.load('test.ann')
    sumCorrect = 0
    for x in xrange(len(testI)):
        if x % 100 == 0:
            print(x)
        vec = []
        for y in testI[x]:
            for z in y:
                vec.append(z)
        guess = u.get_nns_by_vector(vec, 1)
        while isinstance(guess, list):
            guess = guess[0]

        if trainL[guess] == testL[x]:
            sumCorrect = sumCorrect + 1
        else:
            print("wrong! {} != {}".format(trainL[guess], testL[x]))
    print("{}/10000 correct!".format(sumCorrect))
    return sumCorrect
    

def makeIndex(label, img):
    f = 784
    t = AnnoyIndex(f)
    
    for x in xrange(len(img)):
        if x % 1000 == 0:
            print(x)
        vec = []
        for y in img[x]:
            for z in y:
                vec.append(z)
        t.add_item(x, vec)
    t.build(10)
    t.save('test.ann')
    
def get_test():
    with open("t10k-labels-idx1-ubyte", "rb") as fl:
        mag = struct.unpack(">I",fl.read(4))
        n = struct.unpack(">I", fl.read(4))
        labels = np.fromfile(fl, dtype=np.int8)
    
    with open("t10k-images-idx3-ubyte", "rb") as fi:
        magicNumber = struct.unpack(">I", fi.read(4))
        numImages = struct.unpack(">I", fi.read(4))[0]
        rows = struct.unpack(">I", fi.read(4))[0]
        cols = struct.unpack(">I", fi.read(4))[0]
        img = np.fromfile(fi, dtype=np.uint8).reshape(10000, rows, cols)

    return labels, img        
        

def get_train():
    with open("train-labels-idx1-ubyte", "rb") as fLabel:
        magicAndNum = fLabel.read(8)
        unpackedNum, n = struct.unpack(">II", magicAndNum)
        labels = np.fromfile(fLabel, dtype=np.int8)
    with open("train-images-idx3-ubyte", "rb") as f:
        magicNumber = struct.unpack(">I", f.read(4))
        numImages = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        img = np.fromfile(f, dtype=np.uint8).reshape(60000, rows, cols)
    return labels, img
    

    
if __name__ == "__main__":
    main()
    

