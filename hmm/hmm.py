#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import copy
import re
import codecs
import pickle
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

DEFAULT_PROBABILITY = 0.000000001
observation_matrix = {}
transition_matrix = {}
pi_state = {}
state_set = set()
observation_set = set()
data_path = "199801人民日报.data"
model_path = "hmm.model"


def read_data(filename):
    """读取训练数据"""
    sentences = []
    sentence = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f.readlines():
            word_label = line.strip().split("\t")
            if len(word_label) == 2:
                observation_set.add(word_label[0])
                state_set.add(word_label[1])
                sentence.append(word_label)
            else:
                sentences.append(sentence)
                sentence = []
    return sentences


def train():
    """训练函数"""
    print("begin training...")
    sentences = read_data(data_path)
    for sentence in sentences:
        pre_label = -1
        for word, label in sentence:
            observation_matrix[label][word] = observation_matrix.setdefault(
                label, {}).setdefault(word, 0) + 1
            if pre_label == -1:
                pi_state[label] = pi_state.setdefault(label, 0) + 1
            else:
                transition_matrix[
                    pre_label][label] = transition_matrix.setdefault(
                        pre_label, {}).setdefault(label, 0) + 1

            pre_label = label

    #归一化
    for key, value in transition_matrix.items():
        number_total = 0
        for k, v in value.items():
            number_total += v
        for k, v in value.items():
            transition_matrix[key][k] = 1.0 * v / number_total

    for key, value in observation_matrix.items():
        number_total = 0
        for k, v in value.items():
            number_total += v
        for k, v in value.items():
            observation_matrix[key][k] = 1.0 * v / number_total

    number_total = sum(pi_state.values())
    for k, v in pi_state.items():
        pi_state[k] = 1.0 * v / number_total

    print("finish training...")
    save_model()


def predict():
    """模型预测"""
    #model = load_model()
    text = u"我在贝壳网做自然语言处理"
    min_probability = -1 * float("inf")
    words = [{} for _ in text]
    path = {}  #保存路径
    for state in state_set:  #初始状态
        words[0][state] = 1.0 * pi_state.get(
            state, DEFAULT_PROBABILITY) * observation_matrix.get(
                state, {}).get(text[0], DEFAULT_PROBABILITY)
        path[state] = [state]

    for t in range(1, len(text)):
        new_path = {}
        for state in state_set:
            max_probability = min_probability
            max_state = ""
            for pre_state in state_set:
                probability = words[t-1][pre_state] * transition_matrix.get(pre_state,{}).get(state,DEFAULT_PROBABILITY) \
                * observation_matrix.get(state,{}).get(text[t],DEFAULT_PROBABILITY)
                max_probability, max_state = max((max_probability, max_state),
                                                 (probability, pre_state))
            words[t][state] = max_probability
            tmp = copy.deepcopy(path[max_state])
            tmp.append(state)
            new_path[state] = tmp
        path = new_path
    max_probability, max_state = max(
        (words[len(text) - 1][s], s) for s in state_set)
    print path[max_state]
    result = []
    p = re.compile(u"BM*E|S")
    for i in p.finditer("".join(path[max_state])):
        start, end = i.span()
        word = text[start:end]
        print word
        result.append(word)


def save_model():
    """保存模型"""
    print("saveing model...")
    model = [
        transition_matrix, observation_matrix, pi_state, state_set,
        observation_set
    ]
    with codecs.open(model_path, "wb") as f:
        pickle.dump(model, f)


def load_model():
    """加载模型"""
    print("loading model...")
    with codecs.open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    train()
    predict()
