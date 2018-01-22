# -*- coding: utf-8 -*-

import os
import json
import math
import random
import numpy as np
from concurrent import futures


class BatchGenerator:
    def __init__(self, source_dir, batch_size, max_word_length, answer_key="assignee", sentence_limit=5000, num_valid=100):
        self.source = source_dir
        self.batch_size = batch_size
        self.answer_key = answer_key
        self.sentence_limit = sentence_limit
        self.max_word_length = max_word_length
        self.train_data, self.answer_dict = self.prepare()
        # calculate character embed size
        unique_chars = set()
        for item in self.train_data:
            unique_chars.update(item[1])
        self.chars_dict = {i: x for x, i in enumerate(unique_chars)}
        random.shuffle(self.train_data)
        self.valid_data = self.train_data[:num_valid]
        del self.train_data[:num_valid]
        self.num_classes = len(self.answer_dict)
        self.num_batches = int(math.ceil(len(self.train_data) / self.batch_size))
        self.num_valids = int(math.ceil(len(self.valid_data) / self.batch_size))
        print("Train data: {}, Validation data: {}, num_classes: {}".format(len(self.train_data), len(self.valid_data), self.num_classes))

    def batches(self):
        random.shuffle(self.train_data)
        batch_x, batch_y = list(), list()
        for item in self.train_data:
            batch_y.append(self.answer_dict[item[0]])
            x_data = np.split(np.asarray([self.chars_dict[char] for char in item[1]] + [0] * (self.sentence_limit - len(item[1])),
                                         dtype="int32"), self.sentence_limit//self.max_word_length)
            batch_x.append(x_data)
            if len(batch_x) >= self.batch_size:
                yield batch_x, batch_y
                batch_x.clear()
                batch_y.clear()
        if len(batch_x) > 0:
            yield batch_x, batch_y

    def valid_batches(self):
        random.shuffle(self.valid_data)
        batch_x, batch_y = list(), list()
        for item in self.valid_data:
            batch_y.append(self.answer_dict[item[0]])
            x_data = np.split(np.asarray([self.chars_dict[char] for char in item[1]] + [0] * (self.sentence_limit - len(item[1])),
                                         dtype="int32"), self.sentence_limit//self.max_word_length)
            batch_x.append(x_data)
            if len(batch_x) >= self.batch_size:
                yield batch_x, batch_y
                batch_x.clear()
                batch_y.clear()
        if len(batch_x) > 0:
            yield batch_x, batch_y

    def prepare(self):  # 모든 파일 읽으면서 데이터 변환해서 train/eval 나눠서 저장
        result, answers = list(), set()
        with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            to_do = [executor.submit(self.convert, json_path) for json_path in self.read_json()]
            for future in futures.as_completed(to_do):
                data, answer_set = future.result()
                result += data
                answers.update(answer_set)
            answer_dict = dict()
            for i, answer in enumerate(answers):
                answer_dict[answer] = i
        return result, answer_dict

    def convert(self, json_file):
        result, answer_set = list(), set()
        for item in json.load(open(json_file)):
            sentence = ", ".join(["[" + k + "] : " + item[k] for k in item.keys() if k != self.answer_key])[:self.sentence_limit]
            answer = item[self.answer_key]
            answer_set.add(answer)
            result.append((answer, sentence))
        return result, answer_set

    def read_json(self):
        return [os.path.join(path, file_name) for (path, _, files) in os.walk(self.source) for file_name in files if file_name[-4:] == "json"]
