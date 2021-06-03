import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel    

class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)
        
        
        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        
        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds

class LSTMATTN(nn.Module):

    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)            
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds

class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device
        self.dr_rate = args.drop_out

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(self.args.n_test, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag, self.hidden_dim//3)
        self.embedding_timediff = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_qclass = nn.Embedding(self.args.n_class, self.hidden_dim//3)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        
        self.embedding_test_exp = nn.Embedding(2, self.hidden_dim)
        self.embedding_question_exp = nn.Embedding(2, self.hidden_dim)
        
        self.embedding_cont = nn.Linear(9, self.hidden_dim)
        
        # embedding combination projection
        self._mask_comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)
        
        self._nonseq_cat_comb_proj = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self._nonseq_comb_proj = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        
        self._seq_comb_proj = nn.Linear((self.hidden_dim//3)*6, self.hidden_dim)
        
        self.emb_drop_out = nn.Dropout(0.2)
        
        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len          
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)
        
#         self.causalconv = CausalConvModel()
        
        self.comb_proj = nn.Linear(self.args.hidden_dim*2, self.args.hidden_dim)
        self.drop_out = nn.Dropout(self.dr_rate)
        self.act = nn.Tanh()
        
        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)
        
        self.activation = nn.Sigmoid()
    
    def forward(self, input):
        last_test, last_question, last_tag, last_qclass, testid_exp, assessmentItemID_exp, cont_feature, test, question, tag, time_diff, qclass, _, mask, interaction = input
        
        batch_size = interaction.size(0)
        
        # 신나는 embedding
        embed_last_test = self.embedding_test(last_test)
        embed_last_question = self.embedding_question(last_question)
        embed_last_tag = self.embedding_tag(last_tag)
        embed_last_qclass = self.embedding_qclass(last_qclass)
        
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_timediff = self.embedding_timediff(time_diff)
        embed_qclass = self.embedding_qclass(qclass)
        embed_interaction = self.embedding_interaction(interaction)
        
        embed_test_exp = self.embedding_test_exp(testid_exp)
        embed_question_exp = self.embedding_question_exp(assessmentItemID_exp)
        
        embed_cont = self.embedding_cont(cont_feature)
        
        # embedding combination projection 
        embed_mask_ = torch.cat([embed_last_test,
                                 embed_last_question,
                                 embed_last_tag,
                                 embed_last_qclass], 1)
        
        embed_nonseq_cat_ = torch.cat([embed_test_exp,
                                       embed_question_exp], 1)
        embed_nonseq_cat = self._nonseq_cat_comb_proj(embed_nonseq_cat_)
        
        embed_nonseq_ = torch.cat([embed_nonseq_cat,
                                   embed_cont], 1)
        
        embed_seq_ = torch.cat([embed_test,
                                embed_question,
                                embed_tag,
                                embed_timediff,
                                embed_qclass,
                                embed_interaction], 2)
        
        embed_mask = self._mask_comb_proj(embed_mask_)
        embed_mask = self.activation(embed_mask) # 0 ~ 1 -> weight 0% ~ 100%
        
        embed_nonseq = self._nonseq_comb_proj(embed_nonseq_)
        embed_seq = self._seq_comb_proj(embed_seq_)
        
        encoded_layers = self.encoder(inputs_embeds=embed_seq, attention_mask=mask)
        bert_out = encoded_layers[0]
        bert_out = bert_out.contiguous().view(batch_size, -1, self.hidden_dim)
        
#         conv_input = bert_out.unsqueeze(1).transpose(2, 3)
#         conv_out = self.causalconv(conv_input).squeeze()
#         out = torch.cat([conv_out, embed_nonseq], 1)
        
        bert_out = bert_out[:, -1]
        out = torch.cat([bert_out, embed_nonseq], 1)
        out = self.comb_proj(out) # -무한대 ~ +무한대
        out = embed_mask * out # 1인 부분만 값이 남음
        
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)
        
        return preds

class CausalConvModel(nn.Module):

    def __init__(self):
        super(CausalConvModel, self).__init__()
        self.conv1 = self.conv2d(1, 45, (5,3), stride=(1,1), padding=(2,0))
        self.conv2 = self.conv2d(45, 45, (4,1), stride=(2,1), padding=(1,0))
        self.conv3 = self.sepConv2d(45, 90, (5,3), stride=(1,1), padding=(2,0))
        self.conv4 = self.conv2d(90, 90, (4,1), stride=(2,1), padding=(1,0))
        self.conv5 = self.sepConv2d(90, 90, (5,3), stride=(1,1), padding=(2,0))
        self.conv6 = self.conv2d(90, 90, (4,1), stride=(2,1), padding=(1,0))
        self.conv7 = self.sepConv2d(90, 90, (5,3), stride=(1,1), padding=(2,0))
        self.deconv1 = self.convTranspose2d(90, 90, (4,1), stride=(2,1), padding=(1,0))
        self.conv8 = self.sepConv2d(180, 90, (5,3), stride=(1,1), padding=(2,0))
        self.deconv2 = self.convTranspose2d(90, 90, (4,1), stride=(2,1), padding=(1,0))
        self.conv9 = self.sepConv2d(180, 90, (5,3), stride=(1,1), padding=(2,0))
        self.deconv3 = self.convTranspose2d(90, 45, (4,1), stride=(2,1), padding=(1,0))
        self.conv10 = self.sepConv2d(90, 45, (5,3), stride=(1,1), padding=(2,0))
        self.conv11 = torch.nn.Conv2d(45, 1, (1,1), stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.deconv1(x7)
        x8 = torch.cat((x5[:,:,:,2:],x8),dim=1)
        x9 = self.conv8(x8)
        x10 = self.deconv2(x9)
        x10 = torch.cat((x3[:,:,:,6:],x10),dim=1)
        x11 = self.conv9(x10)
        x12 = self.deconv3(x11)
        x12 = torch.cat((x1[:,:,:,10:],x12),dim=1)        
        x13 = self.conv10(x12)
        x14 = self.conv11(x13)
        return x14
    
    def sepConv2d(self, in_channels, out_channels, kernel_size, stride, padding):
        return torch.nn.Sequential(torch.nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
                                   torch.nn.BatchNorm2d(in_channels),
                                   torch.nn.PReLU(),
                                   torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), bias=False),
                                   torch.nn.BatchNorm2d(out_channels),
                                   torch.nn.PReLU())
    
    def conv2d(self, in_channels, out_channels, kernel_size, stride, padding):
        return torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                                   torch.nn.BatchNorm2d(out_channels),
                                   torch.nn.PReLU())

    def convTranspose2d(self, in_channels, out_channels, kernel_size, stride, padding):
        return torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                                   torch.nn.BatchNorm2d(out_channels),
                                   torch.nn.PReLU())