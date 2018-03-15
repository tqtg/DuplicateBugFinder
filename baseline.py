import torch
import torch.nn as nn
import torch.nn.functional as F

class textCNN(torch.nn.Module):
    def __init__(self, args):
        super(textCNN, self).__init__()
        self.word_embed = nn.Embedding(args.n_words, args.word_dim)
        self.char_embed = nn.Embedding(args.n_chars, args.char_dim)

        self.word_conv3 = nn.Conv1d(args.word_dim, args.n_filters, 3)
        self.word_conv4 = nn.Conv1d(args.word_dim, args.n_filters, 4)
        self.word_conv5 = nn.Conv1d(args.word_dim, args.n_filters, 5)

        self.char_conv3 = nn.Conv1d(args.char_dim, args.n_filters, 3)
        self.char_conv4 = nn.Conv1d(args.char_dim, args.n_filters, 4)
        self.char_conv5 = nn.Conv1d(args.char_dim, args.n_filters, 5)

        self.pool = nn.MaxPool1d(args.n_filters) 

    def forward(self,x):
        w_emb = self.word_embed(x[0]).transpose(1,2)
        c_emb = self.char_embed(x[1]).transpose(1,2)

        w_conv3 = F.relu(self.word_conv3(w_emb))
        w_conv3 = F.max_pool1d(w_conv3, kernel_size = w_conv3.size()[-1])
        w_conv4 = F.relu(self.word_conv4(w_emb))
        w_conv4 = F.max_pool1d(w_conv4, kernel_size = w_conv4.size()[-1])
        w_conv5 = F.relu(self.word_conv5(w_emb))
        w_conv5 = F.max_pool1d(w_conv5, kernel_size = w_conv5.size()[-1])


        c_conv3 = F.relu(self.char_conv3(c_emb))
        c_conv3 = F.max_pool1d(c_conv3, kernel_size = c_conv3.size()[-1])
        c_conv4 = F.relu(self.char_conv4(c_emb))
        c_conv4 = F.max_pool1d(c_conv4, kernel_size = c_conv4.size()[-1])
        c_conv5 = F.relu(self.char_conv5(c_emb))
        c_conv5 = F.max_pool1d(c_conv5, kernel_size = c_conv5.size()[-1])

        w_features = torch.cat([w_conv3,w_conv4,w_conv5],-2)[:,:,0]
        c_features = torch.cat([c_conv3,c_conv4,c_conv5],-2)[:,:,0]
        tmp = w_features + c_features
        return tmp

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.short_desc_CNN = textCNN(args)
        self.long_desc_CNN = textCNN(args)

        self.prop_MLP = nn.Sequential(nn.Linear(args.n_prop, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU())
        self.projection = nn.Linear(args.n_filters*6 + 128, 128)
    def forward(self,x):
        # x = [info, short desc, long desc]
        info = x['info']
        short_desc = x['short_desc']
        long_desc = x['long_desc']
        
        short_desc_feature = self.short_desc_CNN(short_desc)
        long_desc_feature = self.long_desc_CNN(long_desc)
        prop_feature = self.prop_MLP(info.float())
        
        feature = torch.cat([prop_feature, short_desc_feature, long_desc_feature], -1)
        feature = F.relu(self.projection(feature))
        return feature


