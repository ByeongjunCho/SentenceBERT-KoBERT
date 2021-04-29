from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import random
import math
from sentence_transformers_.sentence_transformers import losses
from sentence_transformers_.sentence_transformers import models
from sentence_transformers_.sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers_.sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num_epochs', type=int, default=8, help='give a epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='give a batch size for training')
    parser.add_argument('-s', '--evaluation_step', type=int, default=100,
                        help='step parameter for write dev score and save best model during training')
    parser.add_argument('-r', '--seed', type=int, default=33, help='random seed')
    args = parser.parse_args()

    # random seed 설정
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    train_batch_size = args.batch_size
    model_name = '/KoBERT'
    tail = '/STS/' + model_name.replace("/", "") + '-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = 'D:\KoBERT_training/output' + tail
    word_embedding_model = models.Transformer('monologg/kobert')

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logging.info("Read STS train dataset")

    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    train_samples = []

    with open('../data/KorNLUDatasets/KorSTS/tune_train.tsv', "rt", encoding="utf-8") as fIn:
        lines = fIn.readlines()
        for line in lines:
            s1, s2, score = line.split('\t')
            score = score.strip()
            score = float(score) / 5.0
            train_samples.append(InputExample(texts=[s1, s2], label=score))

    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    #Read STSbenchmark dataset and use it as development set
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = []

    with open('../data/KorNLUDatasets/KorSTS/tune_dev.tsv', 'rt', encoding='utf-8') as fIn:
        lines = fIn.readlines()
        for line in lines:
            s1, s2, score = line.split('\t')
            score = score.strip()
            score = float(score) / 5.0
            dev_samples.append(InputExample(texts= [s1,s2], label=score))

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

    num_epochs = args.num_epochs

    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    writer = SummaryWriter('./runs/' + tail)
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs=num_epochs,
              evaluation_steps=args.evaluation_step,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              writer=writer
              )


    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    test_samples = []
    with open('../data/KorNLUDatasets/KorSTS/tune_test.tsv', 'rt', encoding='utf-8') as fIn:
        lines = fIn.readlines()
        for line in lines:
            s1, s2, score = line.split('\t')
            score = score.strip()
            score = float(score) / 5.0
            test_samples.append(InputExample(texts=[s1,s2], label=score))

    print("\n\n\n")
    print("======================TEST===================")
    print("\n\n\n")
    model = SentenceTransformer(model_save_path)
    print(f"model save path > {model_save_path}")
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
    test_evaluator(model, output_path=model_save_path)