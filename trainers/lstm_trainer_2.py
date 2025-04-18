import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils.featureBuilder import FeatureBuilder
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import joblib

class SeqDataset(Dataset):
    """
    Wraps data for pytorch. We turn our numpy arrays X shapw [N, seq_len,n_feat] and 
    y shape(N) into tensors. Turns data into [N,1] so it matches output shape
    """
    def __init__(self,X,y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        """
        Allows torch to know hiow many samples are there
        """
        return len(self.X)
    
    def __getitem__(self, index):
        """
        retrieves index-th sequence and label. Dataloader uses this to build batches
        """
        return self.X[index],self.y[index]


class LSTMReg(nn.Module):
    """
    n_feat : number of features per time step
    hidden: size of each LSTM's hidden state
    layers : how many LSTM layers to stack
    bidir : if true we use bidirectional LSTM reads the sequence forward and backward. Bidirectional means tqo LSTM runs in parallel 
    one left to right one right to left Outputs gets concatenated, si we pature patterns that might depend on future as well as past context. 
    """
    def __init__(self, n_feat , hidden = 64,layers =2,bidir = True):
        super().__init__()
        self.lstm  = nn.LSTM(n_feat, 
                             hidden,
                             layers, 
                             batch_first=True,# data is shaped (batch,seq,feature)
                            dropout=0.1, # 10% dropout between layers
                            bidirectional=bidir)
        
        self.norm = nn.LayerNorm(hidden*(2 if bidir else 1)) # Normalizes the last hidden output vector per sample to stabilize and speed up training
        
        ########### Regression head
        self.head = nn.Sequential(nn.Dropout(0.2), # random 20% drop for regularization
                                  nn.Linear(hidden*(2 if bidir else 1),32), #project to 32 dims
                                  nn.Tanh(), # non linear activation
                                  nn.Linear(32,1) # final regression output
                                  )
        """
        A small feed forward network on top of the LSTM. Takes the normalized hidden vector
        shrinks to 32. Applies a tanh and outputs 1 value (predicted log return)
        """
        
    def forward(self,x):
        """
        out : Tensor of shape [batc, seq_len;hidden*directions]
        We ignore hidden cell/states because we jsut want sequence outputs  
        """
        out, _ = self.lstm(x)
        out = self.norm(out[:,-1]) # picks last timestamp for each sequence in the batch all the final hidden features
        return self.head(out) # pass through MLP head 
    
    
#---------------------- Read csv and build features------------------------


data = pd.read_csv("/Users/nazaraktosun/CryptoCapade/trainers/sample_data/BTC-USD_data.csv", skiprows=[1])
data.rename(columns={'Price': 'date'}, inplace=True)
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', errors='coerce')
data.set_index('date', inplace=True)

cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in cols_to_convert:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
    else:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data["Log Returns"] = np.log(data["Close"] / data["Close"].shift(1))
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=["Log Returns"], inplace=True)

fb = FeatureBuilder(data, target_col= 'Log Returns', n_lags=5)
X,y = fb.get_features_and_target()

SEQ = 10
X_seq , y_seq = [],[]
for i in range(SEQ,len(X)):
    X_seq.append(X.iloc[i-SEQ:i].values)
    y_seq.append(y.iloc[i])

X_seq,y_seq = np.array(X_seq,np.float32), np.array(y_seq,np.float32)

spl = int(len(X_seq)*0.8)

X_train, X_val = X_seq[:spl], X_seq[spl:]
y_train,y_val = y_seq[:spl], y_seq[spl:]

scaler = StandardScaler().fit(X_train.reshape(-1,X_train.shape[-1]))

# Reshape X_train to 2D for scaler
X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
# Apply scaler
X_train_scaled = scaler.transform(X_train_reshaped)
# Reshape back to 3D
X_train = X_train_scaled.reshape(X_train.shape)

# Repeat for X_val
X_val_reshaped = X_val.reshape(-1, X_val.shape[2])
X_val_scaled = scaler.transform(X_val_reshaped)
X_val = X_val_scaled.reshape(X_val.shape)

joblib.dump(scaler,'scaler.gz')

train_dl = DataLoader(SeqDataset(X_train,y_train),32, shuffle= False)
val_dl = DataLoader(SeqDataset(X_val,y_val),32,shuffle = False)

#------------------- Train-------------------------

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMReg(n_feat= X_train.shape[-1]).to(dev)
opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5, patience=4, verbose=True)
lossf = nn.MSELoss()

best,patience, cnt = np.inf,10,0

for ep in range(1,101):
    model.train()
    for xb,yb in train_dl:
        xb,yb = xb.to(dev) , yb.to(dev)
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
        opt.step()
        
    model.eval()
    with torch.no_grad():
        vloss = np.mean([lossf(model(xb.to(dev)), yb.to(dev)).item()
                for xb, yb in val_dl])
        
    sched.step(vloss)
    
    print(f"Epoch {ep:03d} | val {vloss:.5e}")
    
    if vloss+1e-6 < best:
        best = vloss; cnt =0
        torch.save(model.state_dict(), 'best.pt')
    else:
        cnt +=1
        if cnt >= patience:
            print("Early stop");break
            
            
#==================Evaluation=============================
model.load_state_dict(torch.load('best.pt'))
model.eval()    
with torch.no_grad():
    preds = model(torch.tensor(X_val).to(dev)).cpu().numpy().flatten()

rmse = np.sqrt(mean_squared_error(y_val,preds))
mae = mean_squared_error(y_val,preds)
dir_acc = np.mean(np.sign(preds) == np.sign(y_val))
print(f"\nRMSE {rmse:.3e} | MAE {mae:.3e} | DirAcc {dir_acc:.2%}")

plt.figure(figsize=(10,4))
plt.plot(y_val[-150:], label="true")
plt.plot(preds[-150:], label="pred")
plt.legend(); plt.tight_layout(); plt.show()


# ---------------- Next‑day prediction ---------------
last_block = X.iloc[-SEQ:]
seq = scaler.transform(last_block.values).reshape(1, SEQ, -1)
with torch.no_grad():
    next_ret = model(torch.tensor(seq, dtype=torch.float32).to(dev)).item()

last_close = data["Close"].iloc[-1]
print(f"\nNext log‑return {next_ret:+.5f}")
print(f"Predicted next Close ${last_close*np.exp(next_ret):,.2f}")