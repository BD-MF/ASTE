# encoding=utf-8

import numpy as np


Triplet_End = '</s>'
Inner_Interval = '<d>'
Sentence_Start = '<e>'
Sentence_End = '</e>'

# <e >
# food < d > good < d > pos < /s >
# ...
# </e >


def prepro_generator(file_name):
    print("开始处理数据:", file_name)
    fin = open(file_name, 'r', encoding='utf-8')
    lines = fin.readlines()
    fout = open(file_name.split('.')[0]+'.txt', 'w', encoding='utf-8')
    triplet_cnt = 0
    instance_cnt = 0
    polarity_dict = {'NEU': 'neutral', 'POS': 'positive', 'NEG': 'negative'}

    for i in range(0, len(lines), 2):
        instance_cnt += 1
        text = lines[i].strip()
        triplets = lines[i+1].strip().split(';')
        target_text = '<e> '
        for triplet in triplets:
            triplet_cnt += 1
            triplet = eval(triplet)
            aspect_start, aspect_end = triplet[0]
            opinion_start, opinion_end = triplet[1]
            #print(" ".join(text.split()[11:12]))
            aspect = " ".join(text.split()[aspect_start:aspect_end+1])
            opinion = " ".join(text.split()[opinion_start:opinion_end+1])
            polarity_str = polarity_dict[triplet[2]]
            target_text += aspect+' <d> '+opinion+' <d> '+polarity_str+' </s> '
        target_text += '</e>'
        fout.write(text+'\n'+target_text+'\n')


if __name__ == '__main__':
    print("开始处理全部数据，使其格式适合Pointer Generator based Seq2Seq Network")

    file_names = ['datasets/14lap/dev.pair', 'datasets/14lap/train.pair', 'datasets/14lap/test.pair',
                  'datasets/14rest/dev.pair', 'datasets/14rest/train.pair', 'datasets/14rest/test.pair',
                  'datasets/15rest/dev.pair', 'datasets/15rest/train.pair', 'datasets/15rest/test.pair',
                  'datasets/16rest/dev.pair', 'datasets/16rest/train.pair', 'datasets/16rest/test.pair']
    # test_file_name = 'datasets/14rest/dev.pair'

    for file_name in file_names:
        prepro_generator(file_name)


'''
################################
14rest/train.txt
total instance count: 1300
aspect overlap instance count: 247
opinion overlap instance count: 187
aspect overlap triplet count: 276
opinion overlap triplet count: 302
aspect count: 2077
opinion count: 2145
triplet count: 2409
14rest/test.txt
total instance count: 496
aspect overlap instance count: 116
opinion overlap instance count: 77
aspect overlap triplet count: 151
opinion overlap triplet count: 238
aspect count: 849
opinion count: 862
triplet count: 1014
14rest/valid.txt
total instance count: 323
aspect overlap instance count: 50
opinion overlap instance count: 42
aspect overlap triplet count: 58
opinion overlap triplet count: 89
aspect count: 530
opinion count: 524
triplet count: 590
################################
15rest/train.txt
total instance count: 593
aspect overlap instance count: 114
opinion overlap instance count: 37
aspect overlap triplet count: 119
opinion overlap triplet count: 70
aspect count: 834
opinion count: 923
triplet count: 977
15rest/test.txt
total instance count: 318
aspect overlap instance count: 46
opinion overlap instance count: 22
aspect overlap triplet count: 47
opinion overlap triplet count: 24
aspect count: 426
opinion count: 455
triplet count: 479
15rest/valid.txt
total instance count: 148
aspect overlap instance count: 30
opinion overlap instance count: 12
aspect overlap triplet count: 33
opinion overlap triplet count: 29
aspect count: 225
opinion count: 238
triplet count: 260
################################
16rest/train.txt
total instance count: 842
aspect overlap instance count: 151
opinion overlap instance count: 57
aspect overlap triplet count: 157
opinion overlap triplet count: 99
aspect count: 1183
opinion count: 1289
triplet count: 1370
16rest/test.txt
total instance count: 320
aspect overlap instance count: 54
opinion overlap instance count: 23
aspect overlap triplet count: 56
opinion overlap triplet count: 64
aspect count: 444
opinion count: 465
triplet count: 507
16rest/valid.txt
total instance count: 210
aspect overlap instance count: 38
opinion overlap instance count: 14
aspect overlap triplet count: 41
opinion overlap triplet count: 20
aspect count: 291
opinion count: 316
triplet count: 334
################################
14lap/train.txt
total instance count: 920
aspect overlap instance count: 133
opinion overlap instance count: 130
aspect overlap triplet count: 151
opinion overlap triplet count: 214
aspect count: 1283
opinion count: 1265
triplet count: 1451
14lap/test.txt
total instance count: 339
aspect overlap instance count: 59
opinion overlap instance count: 44
aspect overlap triplet count: 64
opinion overlap triplet count: 76
aspect count: 475
opinion count: 490
triplet count: 552
14lap/valid.txt
total instance count: 228
aspect overlap instance count: 49
opinion overlap instance count: 31
aspect overlap triplet count: 56
opinion overlap triplet count: 45
aspect count: 317
opinion count: 337
triplet count: 380
'''
