# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:53:37 2021
密码生成，lstm+ seq2seq方式
@author: admin
"""
# !pip install torch==1.5.0 torchvision==0.5.0

import torch
import torch.nn as nn
import copy
import numpy as np
import matplotlib.pyplot as plt

# !mkdir data
# !curl -L -o data/train.txt https://github.com/brannondorsey/PassGAN/releases/download/data/rockyou-train.txt
def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

def load_dataset(path, max_length, tokenize=False, max_vocab_size=2048, seed = 2020):
    
    lines = []

    with open(path, 'r', encoding="ISO-8859-1") as f:
        for line in f:
            line = line[:-1]
            if tokenize:
                line = tokenize_string(line)
            else:
                line = tuple(line)

            if len(line) > max_length:
                line = line[:max_length]
                continue # don't include this sample, its too long

            # right pad with ` character
            lines.append(line + ( ("`",)*(max_length-len(line)) ) )

    lines = list(set(lines))# Removing duplictes so that no common element in training and validation
    np.random.seed(seed)

    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    # for i in xrange(100):
    #     print filtered_lines[i]

    # print ("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap

class PasswordDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, max_length = 10):
        self.lines, self.charmap, self.inv_charmap = load_dataset(file_path, max_length)
        self.embedding_size = len(self.charmap) # one hot维度

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        tokenized = torch.tensor([self.charmap[c] for c in self.lines[idx]], dtype = torch.int64) #will shift to precomputed if needed to improve speed
        labeled = torch.zeros_like(tokenized)
        labeled[:-1] = tokenized[1:]
        labeled[-1] = tokenized[0] 
        
        one_hot = torch.nn.functional.one_hot(tokenized, num_classes=self.embedding_size) # size=(max_length,vocab_size)
        # one_hot_label = torch.nn.functional.one_hot(labeled,num_classes=2048)

        return one_hot,labeled
    def get_charmap(self):
        return copy.deepcopy(self.charmap)
    
    def get_inv_charmap(self):
        return copy.deepcopy(self.inv_charmap)


class RNNModule(nn.Module):
    def __init__(self, n_vocab,embedding_size=2048, lstm_size=400,num_layers=1):
        super(RNNModule, self).__init__()
        self.n_vocab = n_vocab # 总共不同的字符个数
        self.lstm_size = lstm_size # 隐含层单元个数
        self.num_layers = num_layers # 堆叠的lstm层数
        self.embedding_size = embedding_size
        # self.embedding = nn.Embedding(n_vocab, embedding_size)
        
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=lstm_size,
                            num_layers=num_layers,
                            batch_first=False)
        # print(self.lstm.input_size,self.lstm.hidden_size,self.lstm.num_layers)
        self.drop = nn.Dropout(p=0.5)
        self.dense = nn.Linear(lstm_size, n_vocab)
        
    def forward(self, x, prev_state):
        # embed = self.embedding(x)
        x = x.permute((1,0,2)) # (seq_len, batch, input_size)
        # print('x.shape:{},state_h.shape:{},state_c.shape:{}'.format(x.shape,prev_state[0].shape,prev_state[1].shape))
        output, state = self.lstm(x, prev_state) # output形如 (seq_len, batch, num_directions * hidden_size):
        output = self.drop(output)
        logits = self.dense(output)
    
        return logits, state
    def zero_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.lstm_size),
                torch.zeros(self.num_layers, batch_size, self.lstm_size))
    
def predict(device, net, words,  vocab_to_int, int_to_vocab, pre_len=10,top_k=5):
    net.eval()
    net = net.to(device)
    batch_size =1
    state_h, state_c = net.zero_state(batch_size)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device) # 注意维度
        ix_onehot = torch.nn.functional.one_hot(ix,net.embedding_size)# ( batch,seq_len, input_size)
        ix_onehot = ix_onehot.permute((1,0,2)).float().to(device) # (seq_len, batch, input_size)
        output, (state_h, state_c) = net(ix_onehot, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words = words+int_to_vocab[choice]
    for _ in range(pre_len-1):
        ix = torch.tensor([[choice]],dtype=torch.int64).to(device)
        ix_onehot = torch.nn.functional.one_hot(ix,net.embedding_size)
        ix_onehot = ix_onehot.permute((1,0,2)).float() # (seq_len, batch, input_size)
        output, (state_h, state_c) = net(ix_onehot, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words=words+int_to_vocab[choice]
    return words
    
def train(datapath='./data.txt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PasswordDataset(datapath) # This takes some time to run
    print('one sample maxlen*dim:{},label:{}'.format(dataset[100][0].shape,dataset[100][1])) # 10*2048
    batch_size = 8
    dataTri = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
    
    n_vocab = len(dataset.get_charmap())
    net = RNNModule(n_vocab,embedding_size=dataset.embedding_size,num_layers=1,lstm_size=400)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    iteration = 0
    max_epochs = 50
    loss_all = []
    for epoch in range(max_epochs):
        state_h, state_c = net.zero_state(batch_size)
        
        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in dataTri:
            iteration += 1
            
            # Tell it we are in training mode
            net.train()

            # Transfer data to GPU
            x = x.float().to(device) # (batch,seq_len, embedding_size)
            y = y.to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.permute(1, 2,0), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_value = loss.item()
            loss_all.append(loss_value)
            if iteration%200==0:
                print('Epoch: {}/{}'.format(epoch, max_epochs),
                  'Iteration: {}'.format(iteration),
                  'Loss: {}'.format(loss_value))
            
            initial_words = 'mwk'
            if iteration % 1000 == 0:
                vocab_to_int = dataset.get_charmap()
                int_to_vocab = dataset.get_inv_charmap()
                pre = predict(device, net, initial_words,vocab_to_int, int_to_vocab, pre_len=10,top_k=5)
                torch.save(net.state_dict(),'model-{}.pth'.format(iteration))
                
                
    # plot and save
    export_batch_size=1
    seq_len = 10
    state_hh, state_cc = net.zero_state(export_batch_size)
    x_sample = torch.rand(export_batch_size,seq_len,net.embedding_size).to('cpu') #(batch,seq_len, embedding_size)
    inputT = (x_sample,(state_hh,state_cc))
    net = net.to('cpu')
    torch.onnx.export(net,inputT,'lstm_seq2seq.onnx',
                      input_names=['input', 'h0', 'c0'],
                      output_names=['output', 'hn', 'cn'])     
    plt.figure()
    plt.plot(np.arange(iteration),loss_all)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    
def test(initial_words= 'th',pre_len=7,datapath='./data.txt',modefile='./model-6000.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PasswordDataset(datapath) # This takes some time to run
    n_vocab = len(dataset.get_charmap())
    net = RNNModule(n_vocab,embedding_size=dataset.embedding_size,num_layers=1,lstm_size=400)
    net.load_state_dict(torch.load(modefile))
    net = net.to(device)
    
    vocab_to_int = dataset.get_charmap()
    int_to_vocab = dataset.get_inv_charmap()
    pretxt = predict(device, net, initial_words,vocab_to_int, int_to_vocab, pre_len=pre_len, top_k=5)
    print('predict pass:{}'.format(pretxt))
    
if __name__=='__main__':
    train()
    # test()
