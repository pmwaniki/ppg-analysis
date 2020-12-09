import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv1d(in_channels, out_channels, kernel, stride, padding=kernel//2,bias=False)
        self.cnn2 = nn.Conv1d(out_channels, out_channels, kernel, stride, padding=kernel//2,bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.BatchNorm1d(in_channels)
        self.layer_norm2 = nn.BatchNorm1d(out_channels)


    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        # self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = self.layer_norm(x)
        x = F.relu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class Net(nn.Module):
    def __init__(self,num_classes,n_convs=2,rnn_dim=128,rnn_hidden_dim=128,dropout=0.0):
        super(Net,self).__init__()
        self.cnn=nn.Conv1d(2,32,kernel_size=3,stride=2,padding=1)
        self.dense_layer=nn.Linear(32,rnn_dim)
        self.resnet_layers=nn.Sequential(*[
            ResidualCNN(32,32,kernel=3,stride=1,dropout=dropout)
            for _ in range(n_convs)
        ])
        self.gru_layers=BidirectionalGRU(rnn_dim=rnn_dim,hidden_size=rnn_hidden_dim,dropout=dropout,batch_first=True)
        self.classifier=nn.Sequential(
            nn.Linear(rnn_hidden_dim*2,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self,x):
        out=self.cnn(x)
        out=self.resnet_layers(out)
        out=out.transpose(1,2)
        out=self.dense_layer(out)
        out=self.gru_layers(out)
        out=self.classifier(out[:,-1,:])
        return out

def init_fun(m):
    if isinstance(m,nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

if __name__=="__main__":
    from torchsummary import summary
    self=Net(n_class=2,rnn_dim=24,rnn_hidden_dim=64).cuda()
    summary(self,(2,800))