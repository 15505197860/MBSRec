import numpy as np
from multiprocessing import Process, Queue
import matplotlib as plt


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


#进行目标交互的采集，正负样本的采样
def sample_function(user_train, Beh, Beh_w, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        recency_alpha = 0.5
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32) #用户的行为序列
        pos = np.zeros([maxlen], dtype=np.int32) #与用户行为序列相对应的正样本
        neg = np.zeros([maxlen], dtype=np.int32)  #负样本
        recency = np.zeros([maxlen], dtype=np.float32)
        nxt = user_train[user][-1]  #获取用户最后一个行为
        idx = maxlen - 1   #从后往前，按时间顺序

        ts = set(user_train[user])  #set: 可以快速判断某个行为是否在用户的历史行为中出现过，这在后续生成负样本时会用到
        #两个指针 i 和 nxt
        for i in reversed(user_train[user][:-1]):   #除最后一个的用户行为i 反转（按时间顺序）
            seq[idx] = i   
            pos[idx] = nxt   #从最后一个行为开始
            recency[idx] = recency_alpha**(maxlen-idx) #第idx的行为的时间衰减
            #print('recency[idx]...', recency[idx])
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)  #生成一个负样本
            nxt = i    
            idx -= 1
            if idx == -1: break
        #print(abc)
        seq_cxt = list()  #用户交互的上下文信息
        pos_cxt = list()
        pos_weight = list()
        neg_weight = list()

        for i in seq :
            seq_cxt.append(Beh[(user,i)])   #获取用户u和行为i对应的上下文信息

        for i in pos :
            pos_cxt.append(Beh[(user,i)])

        for i in pos :
            pos_weight.append(Beh_w[(user,i)])  #损失函数中的α
            neg_weight.append(1.0)    #β

        seq_cxt = np.asarray(seq_cxt)  
        pos_cxt = np.asarray(pos_cxt)    
        pos_weight = np.asarray(pos_weight)  


        return (user, seq, pos, neg, seq_cxt, pos_cxt, pos_weight, neg_weight , recency)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, Beh, Beh_w, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      Beh,
                                                      Beh_w,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
