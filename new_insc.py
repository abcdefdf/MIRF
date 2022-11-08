# -*- coding:utf-8 -*-
import random
import math
import sklearn.metrics as metrics
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from generate_sample.generator import Generator
import file_check
from generate_sample.rollout import Rollout
import pandas as pd
from Incremental_RF import randomforest
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
GENERATED_NUM = 9984
POSITIVE_FILE = 'real.data'
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 38
# TOTAL_BATCH = 60
TOTAL_BATCH = 20
# reward_num = 20
reward_num = 2

# Genrator Parameters
g_emb_dim = 32
g_hidden_dim = 64
g_sequence_len = 100

Train_RF_Samples = 10000
benign_txt = './dataset/train/benign.txt'
malicious_txt = './dataset/train/malicious.txt'

alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
            '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']

int_to_char = dict((i, c) for i, c in enumerate(alphabet))
char_to_int = dict((c, i) for i, c in enumerate(alphabet))

feature_list = [str(i) for i in range(36)]

wrong_sample = []


def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    # with open(output_file, 'w') as fout:
    #     for sample in samples:
    #         string = ' '.join([str(s) for s in sample])
    #         fout.write('%s\n' % string)


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversarial training of Generator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        print("GANLoss")
        print(reward)
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1, 1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        print(loss.size(), reward.size())
        loss = loss * reward
        loss = -torch.sum(loss)
        return loss


def real_samples():
    f = open(benign_txt, 'r')
    data = f.read().splitlines()
    f.close()
    data = data[:9984]
    one_hot = []
    for single_data in data:
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in single_data]
        i = len(integer_encoded)
        while i < g_sequence_len:
            integer_encoded.append(0)
            i += 1
        one_hot.append(integer_encoded)
    with open('real.data', 'w') as fout:
        for sample in one_hot:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


def old_predict(discriminator):
    f = open(benign_txt, 'r')
    benign = f.read().splitlines()
    f.close()
    resultlist = random.sample(range(0, len(benign)), 4096)
    bsample = [benign[i] for i in resultlist]
    f = open(malicious_txt, 'r')
    malicious = f.read().splitlines()
    f.close()
    resultlist = random.sample(range(0, len(benign)), 4096)
    msample = [malicious[i] for i in resultlist]
    label = np.array([[0]*len(bsample) + [1]*len(msample)]).T
    bsample.extend(msample)
    # print(bsample, label)
    test = randomforest.str_to_dataframe01(bsample, label)
    result = list(discriminator[0].predict(test.loc[:, feature_list]))
    t_auc = metrics.roc_auc_score(label, result)
    return t_auc


def new_predict(discriminator, increase_data):
    benign = []
    benign_b = file_check.get_no_dot(benign_txt)
    resultlist = random.sample(range(0, len(benign_b)), len(increase_data))
    for i in resultlist:
        benign.append(benign_b[i])
    test_b = randomforest.str_to_dataframe(benign)
    result = list(discriminator[0].predict(test_b.loc[:, feature_list]))
    test_x = increase_data.loc[:, feature_list]
    test_y = increase_data.loc[:, 'label']
    last_y = [0] * len(benign)

    number = [i for i in range(len(test_y))]
    for l in range(len(discriminator)):
        new_test_x = []
        new_test_y = []
        wrong_number = []

        result_y = list(discriminator[l].predict(test_x))
        if l == len(discriminator)-1:
            # np.append(result, result_y)
            # np.append(last_y, test_y)
            result.extend(result_y)
            last_y.extend(test_y)
        for i, j, k in zip(result_y, test_y, number):
            if i != j:
                wrong_number.append(k)
            else:
                result.append(i)
                last_y.append(j)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        for i in wrong_number:
            new_test_x.append(test_x[i])
            new_test_y.append(test_y[i])
        # test_x = new_test_x
        test_x = pd.DataFrame(new_test_x, columns=feature_list)
        test_y = new_test_y
        # test_y = pd.DataFrame(new_test_y, columns=feature_list[-1])

    t_auc = metrics.roc_auc_score(last_y, result)
    return t_auc


def increase_dataset(gen, generated_num, epoch_num):
    samples = []
    for _ in range(int(generated_num / BATCH_SIZE)):
        sample = gen.sample(BATCH_SIZE, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    new_str = []
    for s in samples:
        new_int = []
        for b in s:
            if (int_to_char[b] == ' '):
                continue
            new_int.append(int_to_char[b])
        str1 = ''.join(new_int)
        new_str.append(str1)
    # dataset = randomforest.str_to_dataframe(new_str, file='./exp/exp-'+str(epoch_num)+'.csv')
    dataset = randomforest.str_to_dataframe(new_str, file='no')
    return dataset


def train_discriminator():
    feature_list = [str(i) for i in range(36)]
    # df = pd.read_csv("./Incremental_RF/test.csv")
    df = pd.read_csv("./Incremental_RF/test_10000.csv")
    df = df[df['label'].isin([0, 1])].sample(frac=1, random_state=66).reset_index(drop=True)
    clf = randomforest.RandomForestClassifier(n_estimators=10, random_state=66)
    clf.fit(df.loc[:, feature_list], df.loc[:, 'label'])
    # clf.print_tree('before.gv')
    dis = [clf]
    return dis


def main(model_name, use, cuda):
    random.seed(SEED)
    np.random.seed(SEED)
    auc = []
    old_auc = []

    # Pretrain Discriminator
    discriminator = train_discriminator()
    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)

    # Generate toy data using target lstm
    print('Generating data ...')
    real_samples()

    # Adversarial Training
    rollout = Rollout(generator, 0.7)
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    malicious_new = []
    for total_batch in range(TOTAL_BATCH):
        # ***********************************************************************************************************#
        # Train the generator for one step
        samples = generator.sample(BATCH_SIZE, g_sequence_len)
        # construct the input to the genrator, add zeros before samples and delete the last column
        zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
        if samples.is_cuda:
            zeros = zeros.cuda()
        inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
        targets = Variable(samples.data).contiguous().view((-1,))
        # calculate the reward
        rewards = rollout.get_reward(samples, reward_num, discriminator, alphabet, cuda)
        # rewards = rollout.get_reward(samples, 2, discriminator, alphabet, model_name)
        rewards = Variable(torch.Tensor(rewards))
        if opt.cuda:
            rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))
        prob = generator.forward(inputs)
        loss = gen_gan_loss(prob, targets, rewards)
        gen_gan_optm.zero_grad()
        loss.backward()
        gen_gan_optm.step()
        rollout.update_params()

        df = increase_dataset(generator, 640, total_batch)
        increase_data = df[df['label'].isin([0, 1])].sample(frac=1, random_state=66).reset_index(drop=True)

        t_auc = new_predict(discriminator, increase_data)
        o_auc = old_predict(discriminator)
        # t_auc, dis_loss_i, gen_loss_i, reward_i, b_c, m_c = AUC(generator, discriminator, model_name, 4096)
        print('Epoch[%d]: auc is %f' % (total_batch, t_auc))
        old_auc.append(o_auc)
        auc.append(t_auc)
        print(auc, old_auc)

        # ***********************************************************************************************************#
        # 检测器增量更新
        for dis in discriminator:
            for _, row in increase_data.iterrows():
                dis.increase_sample(row[:-1], row[-1:])
        # 记录分类错误数据
        test_x = increase_data.loc[:, feature_list]
        number = [i for i in range(len(test_x))]
        test_y = increase_data.loc[:, 'label']
        for l in range(len(discriminator)):
            new_test_x = []
            new_test_y = []
            wrong_number = []
            result_y = list(discriminator[l].predict(test_x))
            for m, j, k in zip(result_y, test_y, number):
                if m != j:
                    wrong_number.append(k)
            test_x = np.array(test_x)
            test_y = np.array(test_y)
            for m in wrong_number:
                new_test_x.append(test_x[m])
                new_test_y.append(test_y[m])
            # test_x = new_test_x
            test_y = new_test_y
            test_x = pd.DataFrame(new_test_x, columns=feature_list)
        wrong_sample.extend(test_x)
        print('wrong sample: ', len(wrong_sample))
        # malicious_new = test_x
        if len(wrong_sample) > 1000:
            # 增加分类器
            benign = []
            benign_b = file_check.get_no_dot(benign_txt)
            resultlist = random.sample(range(0, len(benign_b)), len(malicious_new))
            for i in resultlist:
                benign.append(benign_b[i])
            test_b = randomforest.benign_to_dataframe(benign)
            label = np.array([[1] for i in range(len(malicious_new))])
            print(test_b, label)
            feature = np.column_stack((malicious_new, label))
            print(feature)
            feature_l = [str(i) for i in range(36)]
            feature_l.append('label')
            dataset = pd.DataFrame(feature, columns=feature_l)
            # print(test_b, dataset)
            df = pd.concat([test_b, dataset])
            dis_1 = randomforest.RandomForestClassifier(n_estimators=10, random_state=66)
            dis_1.fit(df.loc[:, feature_list], df.loc[:, 'label'])
            discriminator.append(dis_1)
            wrong_sample.clear()

        # result = discriminator.predict(increase_data.loc[:, feature_list])

        # number = [i for i in range(len(result))]
        # wrong_number = []
        # for i, j, k in zip(result, increase_data.loc[:, 'label'], number):
        #     if (i != j) and j == 1:
        #         wrong_number.append(k)
        # for i in wrong_number:
        #     malicious_new.append(increase_data[i][:-1])
        # print('malicious_new', malicious_new, len(malicious_new))

        # 更新检测器后效果
        t_auc = new_predict(discriminator, increase_data)
        o_auc = old_predict(discriminator)
        print('Epoch_increase[%d]', t_auc)
        print('Epoch_increase[%d]', o_auc)
        auc.append(t_auc)
        old_auc.append(o_auc)
        print(auc, old_auc)
    samples_1 = []
    for _ in range(int(9984 / BATCH_SIZE)):
        sample = generator.sample(BATCH_SIZE, g_sequence_len).cpu().data.numpy().tolist()
        samples_1.extend(sample)
    new_str = []
    for s in samples_1:
        new_int = []
        for b in s:
            if (int_to_char[b] == ' '):
                continue
            new_int.append(int_to_char[b])
        str1 = ''.join(new_int)
        new_str.append(str1)
    f = open("./Result/RF_lstm.txt", 'a')
    for i in new_str:
        f.write(i + '\n')
    f.close()
    # 训练效果
    # print(auc)
    # f = open("./Result/RF1.txt", 'a')
    # for i in auc:
    #     f.write(str(i) + ' ')
    # f.write('\n')
    # for i in old_auc:
    #     f.write(str(i) + ' ')
    # f.write('\n')
    # f.close()


if __name__ == '__main__':
    main('RF', 'no', opt.cuda)
