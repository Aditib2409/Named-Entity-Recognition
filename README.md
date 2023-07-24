# Named-Entity-Recognition
Built Deep Learning models on Named-Entity- Recognition NER. 

## Dataset
Used the CoNNL-2003 corpus to build a neural network for NER. There is additional word embeddings done using Glove word embeddings method.

## Evalation
For evaluation, an official evaluation script is created called `conll03eval`. To use this script, you need to install `perl` and prepare a prediction file in the following format:

idx  word  gold  pred

To execute run the following command in the CLI

perl conll03eval < {prediction file}

## Tasks
### Task-1 Simple Bidirectional LSTM 

Implementing the bidirectional LSTM network with PyTorch. The architecture of the network is:

Embedding → BLSTM → Linear → ELU → classifier

The hyper-parameters of the network are listed in the following table:

embedding dim 100
number of LSTM layers 1
LSTM hidden dim 256 LSTM Dropout 0.33
Linear output dim 128

Trained this simple BLSTM model with the training data on NER with SGD as the optimizer.

### Task-2 Using GloVe word embeddings
Use the GloVe word embeddings to improve the BLSTM in Task 1. The way we use the GloVe word embeddings is straightforward: we initialize the embeddings in our neural network with the corresponding vectors in GloVe. Note that GloVe is case-insensitive, but our NER model should be case-sensitive because capitalization is important information for NER.

## About the repo
1. A model file named blstm1.pt for the trained model in Task 1.
2. A model file named blstm2.pt for the trained model in Task 2.
3. Predictions of both dev and test data from Task 1 and Task 2. Name the file with dev1.out, dev2.out, test1.out and test2.out, respectively. All these files should be in the same format as training data.


