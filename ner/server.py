from pathlib import Path
from tensorflow.contrib import predictor
import sys
LINE = "【质量保证】2019夏装新款女装潮韩版中长款印花连衣裙女时尚修身"
if __name__ == '__main__':
    export_dir = '{}/saved_model'.format(sys.argv[1])
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    words = [w.encode() for w in list(LINE.strip())]
    nwords = len(words)
    predictions = predict_fn({'words': [words], 'nwords': [nwords]})
    with Path("{}/tags.dict.txt".format(sys.argv[1])).open() as f:
        tag_dict = {name_lable.split()[1]: name_lable.split()[0] for name_lable in f}
    pre_tags = predictions["tags"]
    labels_res = []
    words_res = []
    index = 0
    tmp_str = ""
    tmp_lab = ""
    for t in pre_tags[0]:
        t = t.decode()
        if t.endswith("b"):
            tmp_lab = tag_dict[t.replace("b", "")]
            tmp_str = LINE.strip()[index:index + 1]
        elif t.endswith("i"):
            tmp_str += LINE.strip()[index:index + 1]
            if index <= len(LINE) - 1:
                words_res.append(tmp_str)
                labels_res.append(tmp_lab)
        else:
            words_res.append(LINE.strip()[index:index + 1])
            labels_res.append(tag_dict[t.replace("o", "")])
        index += 1
    print(",".join([w + ":" + l for w, l in  zip(words_res, labels_res)]))