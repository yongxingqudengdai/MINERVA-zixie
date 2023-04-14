#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/agentbelongstoorganization/"
vocab_dir="datasets/data_preprocessed/agentbelongstoorganization/vocab"
total_iterations=500
path_length=3
hidden_size=50
embedding_size=50
batch_size=128
beta=0.05
Lambda=0.05
# 需要使用实体嵌入？
use_entity_embeddings=1
# 需要进行实体嵌入
train_entity_embeddings=1
# 需要进行关系嵌入
train_relation_embeddings=1
base_output_dir="output/agentbelongstoorganization/"
load_model=0
model_load_dir="/home/sdhuliawala/logs/RL-Path-RNN/wn18rrr/edb6_3_0.05_10_0.05/model/model.ckpt"
nell_evaluation=1