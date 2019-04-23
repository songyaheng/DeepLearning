from pathlib import Path
import sys
import random
if __name__ == '__main__':
    v = float(sys.argv[2])
    train = Path("{}/train.txt".format(sys.argv[1])).open("w", encoding="utf-8")
    test = Path("{}/test.txt".format(sys.argv[1])).open("w", encoding="utf-8")
    train_count = 0
    test_count = 0
    with Path("{}/data.txt".format(sys.argv[1])).open("r") as f:
        for line in f:
            p = random.random()
            if p < v:
                train.write(line)
                train_count += 1
            else:
                test.write(line)
                test_count += 1
    train.close()
    test.close()
    print("train data count: {}, test data count: {}".format(train_count, test_count))