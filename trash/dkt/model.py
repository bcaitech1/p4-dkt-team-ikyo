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


class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device
        self.dr_rate = args.drop_out

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # mask model
        self.mask_model = MaskModel(self.args)
        
        # non-seq model
        self.nonseq_model = NonSeqModel(self.args)
        
        # seq model
        self.seq_model = SeqModel(self.args)
        
        # final layer
        self.batch_norm_comb = nn.BatchNorm1d(self.args.hidden_dim*(5+1))
        self.comb_proj = nn.utils.weight_norm(nn.Linear(self.args.hidden_dim*(5+1), self.args.hidden_dim))
        self.drop_out = nn.Dropout(self.dr_rate)
        self.act = nn.Tanh()
        
        # Fully connected layer
        self.fc = nn.utils.weight_norm(nn.Linear(self.args.hidden_dim, 1))
        
        self.activation = nn.Sigmoid()
    
    def forward(self, input):
        last_test, last_question, last_tag, last_qclass, testid_exp, assessmentItemID_exp, cont_feature, test, question, tag, time_diff, qclass, correct, mask, interaction = input
        
        batch_size = len(interaction)
        
        embed_mask = self.mask_model((last_test, last_question, last_tag, last_qclass))
        nonseq_bert_out = self.nonseq_model((testid_exp, assessmentItemID_exp, cont_feature))
        seq_bert_out = self.seq_model((test, question, tag, qclass, time_diff, interaction, mask))
        
        out = torch.cat([seq_bert_out, nonseq_bert_out], 1) # B, emb*6
        out = self.batch_norm_comb(out)
        out = self.comb_proj(out) # -무한대 ~ +무한대
        out = self.act(out)
        out = embed_mask * out # 1인 부분만 값이 남음
        
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)
        
        return preds

    
class MaskModel(nn.Module):
    
    def __init__(self, args):
        super(MaskModel, self).__init__()
        self.args = args
        self.device = args.device
        self.dr_rate = args.drop_out

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        self.embedding_test = nn.Embedding(self.args.n_test, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag, self.hidden_dim//3)
        self.embedding_qclass = nn.Embedding(self.args.n_class, self.hidden_dim//3)
        
        self.emb_drop_out = nn.Dropout(0.2)
        
        # embedding combination projection
        self.batch_norm_mask = nn.BatchNorm1d((self.hidden_dim//3)*4)
        self._mask_comb_proj = nn.utils.weight_norm(nn.Linear((self.hidden_dim//3)*4, self.hidden_dim))
        
        self.activation = nn.Sigmoid()
    
    def forward(self, input):
        last_test, last_question, last_tag, last_qclass = input
        
        batch_size = len(last_test)
        
        # 신나는 embedding
        embed_last_test = self.embedding_test(last_test)
        embed_last_question = self.embedding_question(last_question)
        embed_last_tag = self.embedding_tag(last_tag)
        embed_last_qclass = self.embedding_qclass(last_qclass)
        
        # embedding combination projection 
        embed_mask_ = torch.cat([embed_last_test,
                                 embed_last_question,
                                 embed_last_tag,
                                 embed_last_qclass], 1)

        ## mask
        embed_mask_ = self.batch_norm_mask(embed_mask_)
        embed_mask = self._mask_comb_proj(embed_mask_)
        embed_mask = self.activation(embed_mask) # 0 ~ 1 -> weight 0% ~ 100%
        return embed_mask

    
class NonSeqModel(nn.Module):
    
    def __init__(self, args):
        super(NonSeqModel, self).__init__()
        self.args = args
        self.device = args.device
        self.dr_rate = args.drop_out

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        self.embedding_test_exp = nn.Embedding(2, self.hidden_dim//3)
        self.embedding_question_exp = nn.Embedding(2, self.hidden_dim//3)
        
        self.batch_norm_total = nn.BatchNorm1d(3)
        self.batch_norm_tag = nn.BatchNorm1d(3)
        self.batch_norm_qclass = nn.BatchNorm1d(3)
        self.embedding_total_cont = nn.utils.weight_norm(nn.Linear(3, self.hidden_dim//3))
        self.embedding_tag_cont = nn.utils.weight_norm(nn.Linear(3, self.hidden_dim//3))
        self.embedding_qclass_cont = nn.utils.weight_norm(nn.Linear(3, self.hidden_dim//3))
        
        # embedding combination projection
        self.batch_norm_nonseq = nn.BatchNorm1d(5)
        self._nonseq_comb_proj = nn.utils.weight_norm(nn.Linear(self.hidden_dim//3, self.hidden_dim))
        
        self.emb_drop_out = nn.Dropout(0.2)
        
        # Bert config
        self.nonseq_config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=1,
        )

        # Defining the layers
        # Bert Layer
        self.nonseq_encoder = BertModel(self.nonseq_config)

        self.act = nn.Tanh()
    
    def forward(self, input):
        testid_exp, assessmentItemID_exp, cont_feature = input
        
        batch_size = len(testid_exp)
        
        # 신나는 embedding
        embed_test_exp = self.embedding_test_exp(testid_exp).unsqueeze(1)
        embed_question_exp = self.embedding_question_exp(assessmentItemID_exp).unsqueeze(1)
        
        embed_total_cont_ = self.batch_norm_total(cont_feature[:,:3])
        embed_tag_cont_ = self.batch_norm_tag(cont_feature[:,3:6])
        embed_qclass_cont_ = self.batch_norm_qclass(cont_feature[:,6:])
        
        embed_total_cont = self.act(self.embedding_total_cont(embed_total_cont_).unsqueeze(1))
        embed_tag_cont = self.act(self.embedding_tag_cont(embed_tag_cont_).unsqueeze(1))
        embed_qclass_cont = self.act(self.embedding_qclass_cont(embed_qclass_cont_).unsqueeze(1))
        
        # embedding combination projection 
        embed_nonseq_ = torch.cat([embed_test_exp,
                                   embed_question_exp,
                                   embed_total_cont,
                                   embed_tag_cont,
                                   embed_qclass_cont], 1)
        
        ## non-seq
        embed_nonseq_ = self.batch_norm_nonseq(embed_nonseq_)
        embed_nonseq = self.act(self._nonseq_comb_proj(embed_nonseq_))
        
        nonseq_mask = torch.ones([batch_size, 5]).to(torch.float32).to(self.device)
        position_ids = torch.zeros([batch_size, 5]).to(torch.int64).to(self.device)
        nonseq_encoded_layers = self.nonseq_encoder(inputs_embeds=embed_nonseq,
                                                    attention_mask=nonseq_mask,
                                                    position_ids=position_ids)
        nonseq_bert_out = nonseq_encoded_layers[0]
        nonseq_bert_out = nonseq_bert_out.contiguous().view(batch_size, -1, self.hidden_dim)
        nonseq_bert_out = torch.flatten(nonseq_bert_out, start_dim=1) # B, emb*5
        return nonseq_bert_out


class SeqModel(nn.Module):
    
    def __init__(self, args):
        super(SeqModel, self).__init__()
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
        
        self.emb_drop_out = nn.Dropout(0.2)
        
        # embedding combination projection
        self.batch_norm_seq = nn.BatchNorm1d(self.args.max_seq_len-1)
        self._seq_comb_proj = nn.utils.weight_norm(nn.Linear((self.hidden_dim//3)*6, self.hidden_dim))
                
        # Bert config
        self.seq_config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len-1,
        )

        # Defining the layers
        # Bert Layer
        self.seq_encoder = BertModel(self.seq_config)

        self.drop_out = nn.Dropout(self.dr_rate)
        self.act = nn.Tanh()
    
    def forward(self, input):
        test, question, tag, qclass, time_diff, interaction, mask = input
        
        batch_size = len(test)
        
        # 신나는 embedding
        embed_test = self.embedding_test(test[:,:-1])
        embed_question = self.embedding_question(question[:,:-1])
        embed_tag = self.embedding_tag(tag[:,:-1])
        embed_qclass = self.embedding_qclass(qclass[:,:-1])
        embed_timediff = self.embedding_timediff(time_diff[:,:-1])
        embed_interaction = self.embedding_interaction(interaction[:,1:])
        
        # embedding combination projection 
        embed_seq_ = torch.cat([embed_test,
                                embed_question,
                                embed_tag,
                                embed_timediff,
                                embed_qclass,
                                embed_interaction], 2)
        
        ## seq
        embed_seq_ = self.batch_norm_seq(embed_seq_)
        embed_seq = self.act(self._seq_comb_proj(embed_seq_))
        
        seq_encoded_layers = self.seq_encoder(inputs_embeds=embed_seq,
                                              attention_mask=mask[:,:-1])
        seq_bert_out = seq_encoded_layers[0]
        seq_bert_out = seq_bert_out.contiguous().view(batch_size, -1, self.hidden_dim)
        seq_bert_out = seq_bert_out[:, -1].squeeze() # B, emb
        return seq_bert_out

###################################################### 안 씀 #####################################################

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