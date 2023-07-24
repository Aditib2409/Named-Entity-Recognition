from torch import dropout
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMner(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size, embeddings_lambda):
        super(BiLSTMner, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = embeddings_lambda()
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.elu = nn.ELU()

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.bilstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
