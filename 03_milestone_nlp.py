#####
# Ross Dahlke
# Identifying Opinion Change in Deliberative Settings
# Stanford University, CS 230: Deep Learning
#####

# This script contains the code for the milestone
# For the milestone, I will be making a variety of baseline models
# These baseline models _will not_ include any linguistic features
# I will focus on linguistic features for the main model

# There will be three models that I will train
# 1. tf-idf with some sort of regression
# 2. RNN with pre-trained sentiment
# 3. LSTM (on text)
# 4. Various transformers

### tf-idf

## going to use spacy's vector similarity functionality
import spacy
nlp = spacy.load("en_core_web_lg")
import docx
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats

# similarities
deltas = pd.read_csv("data/processed/survey/opinion_deltas.csv")
similarity_list = []
delta_list = []
for id in deltas["id"]:
    n_breaks = 10
    try:
        doc = docx.Document("data/processed/transcripts/" + id + ".docx")
        paragraphs = [p.text for p in doc.paragraphs][:-10]
        break_index = round(len(paragraphs) / n_breaks)
        s = " "
        text_0 = nlp(s.join(paragraphs[:break_index]))
        text_1 = nlp(s.join(paragraphs[-break_index * (n_breaks - 1):]))
        similarity = text_0.similarity(text_1)
        similarity_list.append(similarity)
        delta = abs(deltas[deltas["id"] == id]["delta"].values[0])
        delta_list.append(delta)
        print()
        print(id)
        print(similarity)
        print(delta)
    except Exception:
        print("passed on " + id + ". No transcript.")
        pass

# linear regression
similarity_train = np.reshape(np.asarray(similarity_list[0:90]), (-1, 1))
similarity_test = np.reshape(np.asarray(similarity_list[91:101]), (-1, 1))
np.shape(np.reshape(np.asarray(delta_list[0:90]), (-1, 1, 1)))
delta_train = np.reshape(np.asarray(delta_list[0:90]), (-1, 1))
delta_test = np.reshape(np.asarray(delta_list[91:101]), (-1, 1))

model = LinearRegression()

model.fit(similarity_train, delta_train)

delta_test_pred = model.predict(similarity_test)

model.coef_

mean_squared_error(delta_test, delta_test_pred)

r2_score(delta_test, delta_test_pred)

plt.scatter(similarity_test, delta_test, color = "black")
plt.plot(similarity_test, delta_test_pred, color = "blue", linewidth = 3)
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(similarity_list, delta_list)

slope
p_value

## RNN with sentiment
# sentiment analysis based off: https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c
from textblob import TextBlob

polarity_list = []
subjectivity_list = []
delta_list = []
for id in deltas["id"]:
    polarities = []
    subjectivities = []
    try:
        doc = docx.Document("data/processed/transcripts/" + id + ".docx")
        paragraphs = [p.text for p in doc.paragraphs][:-10]
        for paragraph in paragraphs:
            polarity = TextBlob(paragraph).sentiment[0]
            polarities.append(polarity)
            subjectivity = TextBlob(paragraph).sentiment[1]
            subjectivities.append(subjectivity)
        polarity_list.append(polarities)
        subjectivity_list.append(subjectivities)
        delta = abs(deltas[deltas["id"] == id]["delta"].values[0])
        delta_list.append(delta)
    except Exception:
        print("passed on " + id + ". No transcript.")
        pass

# RNN https://www.kaggle.com/purplejester/pytorch-deep-time-series-classification
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader

seed = 1
np.random.seed(seed)

# can do some preprocessing in the future, but it shouldn't be too bad given range from the sentiment models

polarity_list_padded = []
max = 569
for list in polarity_list:
    to_pad = max - len(list)
    polarity_list_padded.append(np.pad(list, (to_pad, 0)))
polarity_array = np.asarray(polarity_list_padded)[:, :, np.newaxis]
polarity_feat = polarity_array.shape[1]

subjectivity_list_padded = []
for list in subjectivity_list:
    to_pad = max - len(list)
    subjectivity_list_padded.append(np.pad(list, (to_pad, 0)))
subjectivity_array = np.asarray(subjectivity_list_padded)[:, :, np.newaxis]
subjectivity_feat = subjectivity_array.shape[1]

delta_array = np.asarray(delta_list)

np.shape(delta_array)
np.shape(subjectivity_array)
np.shape(polarity_array)

def create_datasets(polarity_array, subjectivity_array, target, train_size = 80, valid_pct=0.1, seed=None):
    """Converts NumPy arrays into PyTorch datsets.

    Three datasets are created in total:
        * training dataset
        * validation dataset
        * testing (un-labelled) dataset

    """
    raw = polarity_array
    fft = subjectivity_array
    sz = train_size
    idx = np.arange(sz)
    trn_idx, val_idx = train_test_split(
        idx, test_size=valid_pct, random_state=seed)
    trn_ds = TensorDataset(
        torch.tensor(raw[:sz][trn_idx]).float(),
        torch.tensor(fft[:sz][trn_idx]).float(),
        torch.tensor(target[:sz][trn_idx]).long())
    val_ds = TensorDataset(
        torch.tensor(raw[:sz][val_idx]).float(),
        torch.tensor(fft[:sz][val_idx]).float(),
        torch.tensor(target[:sz][val_idx]).long())
    tst_ds = TensorDataset(
        torch.tensor(raw[sz:]).float(),
        torch.tensor(fft[sz:]).float(),
        torch.tensor(target[sz:]).long())
    return trn_ds, val_ds, tst_ds

def create_loaders(data, bs=10, jobs=0):
    """Wraps the datasets returned by create_datasets function with data loaders."""

    trn_ds, val_ds, tst_ds = data
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    return trn_dl, val_dl, tst_dl

class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(569, 569, kernel, stride, padding=pad, groups=569)
        self.pointwise = nn.Conv1d(569, 569, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.

    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


class Classifier(nn.Module):
    def __init__(self, raw_ni, fft_ni, no, drop=.5):
        super().__init__()
        self.raw = nn.Sequential(
            SepConv1d(   569,  32, 4, 2, 3, drop=drop),
            SepConv1d(    32,  64, 4, 4, 2, drop=drop),
            SepConv1d(    64, 128, 4, 4, 2, drop=drop),
            SepConv1d(   128, 569, 4, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(569, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))

        self.fft = nn.Sequential(
            SepConv1d(   569,  32, 4, 2, 3, drop=drop),
            SepConv1d(    32,  64, 4, 4, 2, drop=drop),
            SepConv1d(    64, 128, 4, 4, 2, drop=drop),
            SepConv1d(   128, 569, 4, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(569, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))

        self.out = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, no))

    def forward(self, t_raw, t_fft):
        raw_out = self.raw(t_raw)
        fft_out = self.fft(t_fft)
        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(t_in)
        return out

# polarity
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

datasets = create_datasets(polarity_array, subjectivity_array, delta_array)

trn_dl, val_dl, tst_dl = create_loaders(datasets)

lr = 0.001
n_epochs = 3000
iterations_per_epoch = len(trn_dl)
num_classes = 2
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2
trn_sz = 80
loss_history = []
acc_history = []

model = Classifier(polarity_feat, subjectivity_feat, num_classes).to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=lr)

print('Start model training')

for epoch in range(1, n_epochs + 1):

    model.train()
    epoch_loss = 0
    for i, batch in enumerate(trn_dl):
        x_raw, x_fft, y_batch = [t.to(device) for t in batch]
        opt.zero_grad()
        out = model(x_raw, x_fft)
        loss = criterion(out, y_batch)
        epoch_loss += loss.item()
        loss.backward()
        opt.step()

    epoch_loss /= trn_sz
    loss_history.append(epoch_loss)

    model.eval()
    correct, total = 0, 0
    for batch in val_dl:
        x_raw, x_fft, y_batch = [t.to(device) for t in batch]
        out = model(x_raw, x_fft)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

    acc = correct / total
    acc_history.append(acc)

    if epoch % base == 0:
        print(f'Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
        base *= step

    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), 'best.pth')
        print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break

print('Done!')
