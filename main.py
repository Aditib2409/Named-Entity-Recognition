import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gzip

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

docstart = '-DOCSTART-'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import warnings
warnings.filterwarnings("ignore")

class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
class Traindata(Dataset):
    def __init__(self, data,labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence = self.data[index]
        label = self.labels[index]
        lengths = len(sentence)

        if self.transform is not None:
            sentence = self.transform(sentence)
            lengths = self.transform(lengths)

        return sentence, label, lengths

class Testdata(Dataset):
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence = self.data[index]
        lengths = len(sentence)
        
        if self.transform is not None:
            sentence = self.transform(sentence)
            lengths = self.transform(lengths)
            
        return sentence,lengths

class BiLSTMTagger(nn.Module):
    
    def __init__(self,embedding_dim,hidden_size,output_dim):
        super(BiLSTMTagger, self).__init__()
        self.num_layers = NUM_LAYERS
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(len(word2idx.keys()), embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size,num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.hidden = nn.Linear(2*hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(output_dim, NUM_CLASSES)

    def init_hidden_layer(self,batch_size):
        return torch.zeros(2*self.num_layers,batch_size,self.hidden_size)

    def init_cells(self,batch_size):
        return torch.zeros(2*self.num_layers,batch_size,self.hidden_size)
        
    def forward(self, x, hidden, cells, seq_lengths):
        x = self.embeddings(x)
        x = pack_padded_sequence(x, seq_lengths, batch_first=True,enforce_sorted=False)
        x, _ = self.bilstm(x, (hidden,cells))
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.elu(x)
        x = self.classifier(x)
        return x

class BiLSTMner(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, classes_num, lambda_embeddings):
        super(BiLSTMner, self).__init__()
        self.word_embeddings = lambda_embeddings()
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, classes_num)
        self.elu = nn.ELU()

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.bilstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def loaddataset(datapath):
    dataset=[]
    with open(datapath,'r') as file:
        for line in file:
            splitlines = line.split()
            if len(splitlines)!=0:
                dataset.append(splitlines)
        file.close()
    return np.array(dataset)

def read_ner_data(path):
    words, tags = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            split = l.split()
            if len(split) == 0:
                continue
            w = split[0]
            if w == docstart:
                continue
            tag = split[-1]
            words.append(w)
            tags.append(tag)
        return words, tags

def getbatched(words, labels, size):
    for i in range(0, len(labels), size):
        yield (words[i:i + size], labels[i:i + size])

def loadembedding(path):
    from tqdm import tqdm
    import numpy as np
    index_embeddings = dict()
    with gzip.open(path, 'r') as f:
        for l in f.readlines():
            v = l.split()
            w = v[0]
            coeffs = np.asarray(v[1:], dtype='float32')
            index_embeddings[w] = coeffs
    return index_embeddings

def tagidxfromscores(scores):
    pred = []
    for index in range(scores.shape[0]):
        pred.append(int(np.argmax(scores[index])))
    return pred

class indexer:
    def __init__(self, eles):
        self._ele2idx = {"<UNKNOWN>": 0}
        for x in eles:
            if x not in self._ele2idx:
                self._ele2idx[x] = len(self._ele2idx)
        self._idx2ele = {v: k for k,v in self._ele2idx.items()}

    def getele2idx_dict(self):
        return self._ele2idx

    def ele2idx(self, ele):
        return self._ele2idx.get(ele, 0)

    def idx2ele(self, idx):
        return self._idx2ele[idx]

    def eles2idx(self, eles):
        return [self.ele2idx(x) for x in eles]

    def idxs2eles(self, idxs):
        return [self.idx2ele(x) for x in idxs]

    def size(self):
        return len(self._ele2idx)

def calculate_recall(table):
    return table[1, 1] / (table[1, 0] + table[1, 1])

def calculate_precision(table):
    return table[1, 1]/(table[1, 1] + table[0, 1])

def calculate_f1score(table, beta=1):
    precision = calculate_precision(table)
    recall = calculate_recall(table)
    return ((1 + beta**2)*precision*recall)/(beta**2 * precision + recall)

class GetMetrics:
    metriclambdas = {
        "Recall": lambda x: calculate_recall(x),
        "Precision": lambda x: calculate_precision(x),
        "F1 score": lambda x: calculate_f1score(x)
    }

    def __init__(self, classes):
        self.dict_metrics = {metric: [] for metric in GetMetrics.metriclambdas.keys()}
        self.pred_table_classes = {eachclass: [np.zeros((2,2))] for eachclass in classes}

    def calculate_metrics(self):
        for metric, calculator in GetMetrics.metriclambdas.items():
            val = [calculator(x[-1]) for _, x in self.pred_table_classes.items()]
            self.dict_metrics[metric].append(np.nanmean(val))

    def collectmetrics(self):
        self.calculate_metrics()
        for key in self.pred_table_classes.keys():
            self.pred_table_classes[key].append(np.zeros((2,2)))

    @staticmethod
    def calculate_pred_table(pred, true, label):
        pred_table = np.zeros((2, 2))
        for i in range(len(pred)):
            if pred[i] != label and true[i] != label:
                pred_table[0][0] += 1
            if pred[i] != label and true[i] == label:
                pred_table[1, 0] += 1
            if pred[i] == label and true[i] == label:
                pred_table[1, 1] += 1
            if pred[i] == label and true[i] != label:
                pred_table[0, 1] += 1
        return pred_table

    def update_predictions(self, pred, true):
        for k in self.pred_table_classes.keys():
            self.pred_table_classes[k][-1] += GetMetrics.calculate_pred_table(pred, true, k)

    def getmetrics(self):
        return self.dict_metrics

class EmbedFabric:
    @staticmethod
    def initialization(mat, unk_vec, word_indexer, dict_embedding):
        for w, i in word_indexer.getele2idx_dict().items():
            if w in dict_embedding.keys():
                mat[i] = dict_embedding[w]
            else:
                mat[i] = unk_vec
        return mat

    EMBEDDINGS = {
        "glove_embeddings": lambda mat, unk_vec, word_indexer, dict_embedding: EmbedFabric.initialization(mat, unk_vec, word_indexer, dict_embedding)
    }

    @staticmethod
    def getlayerembed(word_indexer, dict_embedding, strategy='glove_embeddings'):
        import numpy as np
        import torch
        import torch.nn as nn

        mat_len = word_indexer.size()
        embedding_dim = next(iter(dict_embedding.values())).shape[0]
        weights = np.zeros((mat_len, embedding_dim))
        unk_vec = np.random.randn(100)
        weights = EmbedFabric.EMBEDDINGS[strategy](weights, unk_vec, word_indexer, dict_embedding)
        embedding = nn.Embedding(mat_len, embedding_dim)
        embedding.load_state_dict({'weight': torch.from_numpy(weights)})
        return embedding

# get data padding done
def batchpadding(batch):
    sentences, sentence_tags, len = [], [], []
    for input in batch:
        sentences.append(input[0])
        sentence_tags.append(input[1])
        len.append(input[2])
    pad_sentences = torch.nn.utils.rnn.pad_sequence(sentences,batch_first=True)
    pad_tags = torch.nn.utils.rnn.pad_sequence(sentence_tags,batch_first=True,padding_value=-1)

    return pad_sentences, pad_tags, len

def batchpadding_test(batch):
  sentences, len = [], []
  for input in batch:
      sentences.append(input[0])
      len.append(input[1])
  pad_sentences = torch.nn.utils.rnn.pad_sequence(sentences,batch_first=True)

  return pad_sentences, len

# create Train dataloader
def get_train_dataloader(train_data, batch_size):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,collate_fn=batchpadding,shuffle=True)

    return train_loader

def get_weights(labels):
    class_weights = 1./(np.unique(labels, return_counts=True)[1])
    return class_weights/np.sum(class_weights)


def train_model(model, optimizer, loss_fn,
                dict_data, batch_size, word_indexer,
                tag_indexer, metrics_training, metrics_validation, num_epochs=5):
    dev_words, dev_tags = dict_data['dev']
    pred_tags = []

    def validate_model():
        with torch.no_grad():
            inputs = torch.tensor(word_indexer.eles2idx(dev_words), dtype=torch.long)
            true_vals = tag_indexer.eles2idx(dev_tags)
            tag_scores = model(inputs)
            prediction = tagidxfromscores(tag_scores)
        metrics_validation.update_predictions(prediction, true_vals)
        metrics_validation.collectmetrics()
        for metric in metrics_validation.dict_metrics.keys():
            print(f"{metric} - {metrics_validation.dict_metrics[metric][-1]}")
        for pred in prediction:
            pred_tags.append(tag_indexer.idx2ele(pred))
        print()

    words, tags = dict_data['train']
    batches = list(getbatched(words, tags, batch_size))
    losses = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        running_loss = 0.0
        for sentence, tags in batches:
            model.zero_grad()
            input_sen = torch.tensor(word_indexer.eles2idx(sentence), dtype=torch.long)
            target_tags = torch.tensor(tag_indexer.eles2idx(tags), dtype=torch.long)
            tag_scores = model(input_sen)
            loss = loss_fn(tag_scores, target_tags)
            loss.backward()
            optimizer.step()
            pred = tagidxfromscores(tag_scores.detach().numpy())
            metrics_training.update_predictions(pred, target_tags)
            running_loss += loss.item() * input_sen.size(0)
        metrics_training.collectmetrics()
        epoch_loss = running_loss / len(batches)
        losses.append(epoch_loss)
    validate_model()
    print()

    return model, metrics_training, metrics_validation, losses, pred_tags


def generating_out_files(out_path, ner_preds, og_dataset, datatype):
    with open(out_path,'w') as file:
        for id in range(len(ner_preds)):
          if datatype != 'test':
            if og_dataset[id][0]=='1':
                file.write('\n')
            file.write(f'{og_dataset[id][0]} {og_dataset[id][1]} {og_dataset[id][2]} {ner_preds[id]}\n')
          else:
            if og_dataset[id][0]=='1':
                file.write('\n')
            file.write(f'{og_dataset[id][0]} {og_dataset[id][1]} {ner_preds[id]}\n')



def preprocess_data(X,y,test_size=0):
    transform = None
    if test_size==0:
        return Traindata(X,y, transform)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    train_data = Traindata(X_train,y_train, transform)
    test_data = Traindata(X_test,y_test, transform)
    return train_data,test_data

def preprocess_test(X):
    transform = None
    test_data = Testdata(X, transform)
    return test_data

tr_path = 'hw4/data/train'
dev_path = 'hw4/data/dev'
ts_path = 'hw4/data/test'
embed_path = 'hw4/embeds/glove.6B.100d.gz'
dev1_out_path = 'hw4/dev1.out'
dev2_out_path = 'hw4/dev2.out'
test1_out_path = 'hw4/test1.out'
test2_out_path = 'hw4/test2.out'

glove_embeds = loadembedding(embed_path)

tr_dataset = loaddataset(tr_path)
dev_dataset = loaddataset(dev_path)
ts_dataset = loaddataset(ts_path)

VOCAB = set(list(tr_dataset[:][:,1]))
TAGS = set(list(tr_dataset[:][:,2]))
NUM_CLASSES = len(TAGS)
NUM_LAYERS = 1

tr_words, tr_tags = read_ner_data(tr_path)
dev_words, dev_tags = read_ner_data(dev_path)
ts_words, ts_tags = read_ner_data(ts_path)

data_dictionary = {
    'train': (tr_words, tr_tags),
    'dev': (dev_words, dev_tags),
    'test': (ts_words, ts_tags)
}

pad_tag = 'PAD'
unknown_tag = 'UNK'

word2idx = {}
word2idx['PAD'] = -1
word2idx['UNK'] = 0
for id, w in enumerate(VOCAB):
    word2idx[w] = id+1

idx2word = {value: key for key, value in word2idx.items()}

tag2idx = {}
tag2idx['PAD'] = -1
for id,t in enumerate(TAGS):
    tag2idx[t] = id

idx2tag = {value: key for key, value in tag2idx.items()}

words_indexer = indexer(tr_words)
tags_indexer = indexer(tr_tags)

sentences = tr_dataset[:][:,[0,1]]
dev_sentences = dev_dataset[:][:,[0,1]]
test_sentences = ts_dataset[:][:, [0,1]]
vec_sen, vec_each, vec_etag, vec_tags, vec_tags_np =[], [], [], [], []

for i,(id,w) in enumerate(sentences):
    if id == '1':
        if i!=0:
            vec_sen.append(torch.LongTensor(vec_each))
            vec_tags.append(torch.LongTensor(vec_etag))
            vec_tags_np.append((vec_etag))
            vec_each, vec_etag = [], []
    vec_each.append(word2idx[w])
    vec_etag.append(tag2idx[tr_dataset[i][2]])

sort_seq = torch.LongTensor(list(map(len, vec_sen)))

dev_sentences = dev_dataset[:][:,[0,1]]
vec_sen_dev=[]
vec_each_dev=[]
vec_tags_dev=[]
vec_tag_dev=[]
for i,(index,word) in enumerate(dev_sentences):
    if index == '1':
        if i!=0:
            vec_sen_dev.append(torch.LongTensor(vec_each_dev))
            vec_tags_dev.append(torch.LongTensor(vec_tag_dev))
            vec_each_dev=[]
            vec_tag_dev=[]
            if i==len(dev_sentences)-1:
                vec_each_dev.append(word2idx[word])
                vec_tag_dev.append(tag2idx[dev_dataset[i][2]])
                vec_sen_dev.append(torch.LongTensor(vec_each_dev))
                vec_tags_dev.append(torch.LongTensor(vec_tag_dev))
    if word.lower() in word2idx.keys():
        vec_each_dev.append(word2idx[word.lower()])
        vec_tag_dev.append(tag2idx[dev_dataset[i][2]])
    else:
        vec_each_dev.append(word2idx['UNK'])
        vec_tag_dev.append(0)

test_sentences = ts_dataset[:][:,[0,1]]
vec_sen_test=[]
vec_each_test=[]
vec_tags_test=[]
vec_tag_test=[]
for i,(index,word) in enumerate(test_sentences):
    if index == '1':
        if i!=0:
            vec_sen_test.append(torch.LongTensor(vec_each_test))
            vec_tags_test.append(torch.LongTensor(vec_tag_test))
            vec_each_test=[]
            vec_tag_test=[]
            if i==len(test_sentences)-1:
                vec_each_test.append(word2idx[word])
                vec_sen_test.append(torch.LongTensor(vec_each_test))
    vec_each_test.append(word2idx[w])

embedding_dim = 100
hidden_dim = 256
lr = 1
mu = 0.5
num_epochs = 30
lstm_layers = 1
linear_output_dim = 128
dropout = 0.33
batch_size = 32

bilstm1 = BiLSTMTagger(embedding_dim, hidden_dim, linear_output_dim).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
optimizer = torch.optim.SGD(bilstm1.parameters(),lr=lr, momentum=mu)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

preprocessed_train = preprocess_data(vec_sen, vec_tags)
train_loader = torch.utils.data.DataLoader(preprocessed_train, batch_size=batch_size,collate_fn=batchpadding,shuffle=True)

preprocessed_test = preprocess_test(vec_sen_test)
test_loader = torch.utils.data.DataLoader(preprocessed_test, batch_size=batch_size,collate_fn=batchpadding_test,shuffle=False)

X = vec_sen_dev
y = vec_tags_dev
preprocessed_dev = preprocess_data(X,y)
dev_loader = torch.utils.data.DataLoader(preprocessed_dev, batch_size=batch_size,collate_fn=batchpadding,shuffle=False)

true_labels = tr_dataset[:,2]
weights = get_weights(true_labels)
class_weights = torch.FloatTensor(weights).to(device)

print(f'Training of BiLSTM-1 has begun ------- ')
for epoch in tqdm(range(num_epochs)):
    correct = 0
    train_loss = 0.0

    bilstm1.train() # prepare model for training
    hidden = bilstm1.init_hidden_layer(batch_size).to(device)
    cells = bilstm1.init_cells(batch_size).to(device)
    for data_sen, target_tag, leng in train_loader:
        data_sen = data_sen.squeeze(1).to(device)
        target_tag = target_tag.to(device)
        bilstm1.zero_grad()
        output = bilstm1(data_sen,hidden,cells,leng).to(device)
        output = output.permute((0,2,1))
        loss = criterion(output,target_tag)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*data_sen.size(0)
        for k in range(len(leng)):
            correct+=np.count_nonzero((torch.argmax(output,dim=1)[k][:leng[k]]==target_tag[k][:leng[k]]).cpu())

    train_accuracy = correct/sum(sort_seq)
    train_loss = train_loss/len(train_loader.dataset)
    
    print(f'Epoch: {epoch+1} | Training Accuracy: {correct/sum(sort_seq):.4f} | Training Loss: {train_loss/len(train_loader.dataset):.6f}')
    torch.save(bilstm1.state_dict(), f'hw4/blstm1.pt')

model1 = BiLSTMTagger(embedding_dim, hidden_dim, linear_output_dim).to(device)
model1.load_state_dict(torch.load('hw4/blstm1.pt'))
model1.eval()

# testing on Dev set
def validate_dev_set(dev_loader, model):
    dev_preds =[]
    total=0 
    correct_preds=0
    with torch.no_grad():
        for i,(data_dev_sen, target_dev_tag, leng_dev) in tqdm(enumerate(dev_loader)):

          data_dev_sen, target_dev_tag = data_dev_sen.squeeze(1).to(device), target_dev_tag.to(device)
          hidden = model.init_hidden_layer(batch_size).to(device)
          cells = model.init_cells(batch_size).to(device)
          output_dev = model(data_dev_sen, hidden, cells, leng_dev).to(device)
          output_dev = output_dev.permute((0,2,1))
          total+=sum(leng_dev)
          for j in (range(len(leng_dev))):
              correct_preds += np.count_nonzero((torch.argmax(output_dev,dim=1)[j]==target_dev_tag[j]).cpu())
              for w_id in torch.argmax(output_dev,dim=1)[j][:leng_dev[j]]:
                  dev_preds.append(idx2tag[int(w_id)])
    return correct_preds/total, dev_preds

def predict_test_set(test_loader, model=model1):
  test_preds = []
  total_test = 0
  correct_preds = 0
  with torch.no_grad():
    for i, (data_test_sen, leng_test) in tqdm(enumerate(test_loader)):
      data_test_sen = data_test_sen.squeeze(1).to(device)
      hidden = model.init_hidden_layer(batch_size).to(device)
      cells = model.init_cells(batch_size).to(device)
      output_test = model(data_test_sen, hidden, cells, leng_test).to(device)
      output_test = output_test.permute((0,2,1))
      total_test+=sum(leng_test)
      for j in (range(len(leng_test))):
          # correct_preds += np.count_nonzero((torch.argmax(output_test,dim=1)[j]==target_dev_tag[j]).cpu())
          for w_id in torch.argmax(output_test,dim=1)[j][:leng_test[j]]:
              test_preds.append(idx2tag[int(w_id)])
    return correct_preds, test_preds

print(f'Validating on Dev set -------------')
dev_accuracy, preds_dev = validate_dev_set(dev_loader, model=model1)

generating_out_files(dev1_out_path, preds_dev, dev_dataset, datatype='dev')
print(f'Generated "dev1.out" file -----------')

print(f'Metrics on dev1 set -----------')
acc = accuracy_score(dev_dataset[:, 2], preds_dev)
f1 = f1_score(dev_dataset[:, 2], preds_dev, average='weighted')
recall = recall_score(dev_dataset[:, 2], preds_dev, average='weighted')
precision = precision_score(dev_dataset[:, 2], preds_dev, average='weighted')

print(f'Predicting on test set ------------')
_, preds_test = predict_test_set(test_loader)

generating_out_files(test1_out_path, preds_test, ts_dataset, datatype='test')
print(f'Generated "test1.out" file -----------')
print(f'acc: {acc} | F: {f1} | R: {recall} | P: {precision}')

print(f'************************************************')
print(f'************************************************')
print(f'************************************************')


models = {}
strategy = 'glove_embeddings'
bilstm2 = BiLSTMner(embedding_dim, hidden_dim, tags_indexer.size(),
                    lambda: EmbedFabric.getlayerembed(words_indexer, glove_embeds, strategy))
models[strategy] = bilstm2

true_labels = [x for x in tags_indexer.getele2idx_dict().values()]

trained_models = {}

for name, model in models.items():
    print(f'Training {name} model')
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    metricloader = GetMetrics(true_labels)
    metricvalidator = GetMetrics(true_labels)

    trained_model, train_metrics, dev_metrics, losses, devpreds = train_model(model, 
                                        optimizer,
                                        loss_fn,
                                        data_dictionary,128,
                                        words_indexer,
                                        tags_indexer,
                                        metricloader,
                                        metricvalidator,
                                        num_epochs)
    torch.save(bilstm1.state_dict(), f'hw4/blstm2.pt')

# Testing the model on test seta
test_metrics = GetMetrics(true_labels)
test_preds = []
for name, model in models.items():
    print(f"{name} results on test set:")
    with torch.no_grad():
        inputs = torch.tensor(words_indexer.eles2idx(ts_words), dtype=torch.long)
        true_vals = tags_indexer.eles2idx(ts_tags)
        tag_scores = model(inputs)
        prediction = tagidxfromscores(tag_scores)
        test_metrics.update_predictions(prediction, true_vals)
        test_metrics.collectmetrics()
        for pred in prediction:
            test_preds.append(tags_indexer.idx2ele(pred))

generating_out_files(dev2_out_path, devpreds, dev_dataset, datatype='dev')       
print(f'Generated "dev2.out" file -----------')

generating_out_files(test2_out_path, test_preds, ts_dataset, datatype='test')       
print(f'Generated "test2.out" file -----------')