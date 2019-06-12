"""Script to build words, chars and tags vocab"""
__author__ = "yaheng.song"

from collections import Counter
from pathlib import Path
import itertools
import json
import sys
MINCOUNT = 1

TAGS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
INDEX = ["b", "i", "o"]

def processLine(line):
    return "".join(line.split())

def product(tags):
    r = 2
    new_tags = TAGS
    tag_dict = {}
    while len(tags) > len(new_tags):
        new_tags = list(map(lambda tp: tp[0] + tp[1], itertools.product(TAGS, repeat=r)))
        r = r + 1
    new_tags = sorted(new_tags)
    for i in range(0, len(tags)):
        tag_dict[tags[i]] = new_tags[i]
    return tag_dict

if __name__ == '__main__':
    rootDataPath = sys.argv[1]
    print("the root data path is " + rootDataPath)
    print('Build vocab words and tags (may take a while...)')

    vocab_tags = set()
    # 1.构建TAG数据集 和字数据集
    for name in ['train', 'test']:
        with Path("{}/{}.txt".format(rootDataPath, name)).open() as f:
            for line in f:
                try:
                    word_line = json.loads(line)["data"]
                    for w in word_line:
                        vocab_tags.add(w["tag"])
                except Exception as e:
                    print(e)
                    print("the input line is Exception : " + line)
    # 2. Encoding Tags And Save
    tag_map = product(list(vocab_tags))
    vocab_tags.clear()
    with Path("{}/tags.dict.txt".format(rootDataPath)).open("w", encoding="utf-8") as f:
        for k, v in tag_map.items():
            f.write("{} {}\n".format(k, v))
    # 3. 构建训练和测试数据机
    print("starting to build data (may take a while...)")
    counter_words = Counter()
    vocab_tags = set()
    f_train_chars = Path("{}/train.words.txt".format(rootDataPath)).open("w", encoding="utf-8")
    f_train_tags = Path("{}/train.tags.txt".format(rootDataPath)).open("w", encoding="utf-8")
    f_test_chars = Path("{}/test.words.txt".format(rootDataPath)).open("w", encoding="utf-8")
    f_test_tags = Path("{}/test.tags.txt".format(rootDataPath)).open("w", encoding="utf-8")

    for name in ['train', 'test']:
        with Path("{}/{}.txt".format(rootDataPath, name)).open() as f:
            for line in f:
                try:
                    word_line = json.loads(line)["data"]
                    words = []
                    label = []
                    for w in word_line:
                        word = processLine(w["word"])
                        if len(word) > 0:
                            w_list = list(word)
                            counter_words.update(w_list)
                            words.append(" ".join(w_list))
                            tag = w["tag"]
                            if len(w_list) == 1:
                                tg = "{}{}".format(tag_map[tag], INDEX[2])
                                vocab_tags.add(tg)
                                label.append(tg)
                            else:
                                for i in range(0, len(w_list)):
                                    if i == 0:
                                        tg = "{}{}".format(tag_map[tag], INDEX[0])
                                        vocab_tags.add(tg)
                                        label.append(tg)
                                    elif i == len(w_list) - 1:
                                        tg = "{}{}".format(tag_map[tag], INDEX[1])
                                        vocab_tags.add(tg)
                                        label.append(tg)
                                    else:
                                        tg = "{}{}".format(tag_map[tag], INDEX[1])
                                        vocab_tags.add(tg)
                                        label.append(tg)
                    if len(words) > 0:
                        if name == "train":
                            f_train_chars.write("{}\n".format(" ".join(words)))
                            f_train_tags.write("{}\n".format(" ".join(label)))
                        else:
                            f_test_chars.write("{}\n".format(" ".join(words)))
                            f_test_tags.write("{}\n".format(" ".join(label)))
                except Exception as e:
                    print(e)
                    print("the input line is Exception : " + line)
    f_train_chars.close()
    f_test_chars.close()
    f_train_tags.close()
    f_test_tags.close()

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}
    with Path('{}/vocab.words.txt'.format(rootDataPath)).open('w', encoding="utf-8") as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(len(vocab_words), len(counter_words)))
    vocab_words.clear()
    counter_words.clear()

    # 4. Tags Save
    # Get all tags from the training set
    with Path('{}/vocab.tags.txt'.format(rootDataPath)).open('w', encoding="utf-8") as f:
        for t in sorted(vocab_tags):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))
