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


class SimpleModel(nn.Module):

    def __init__(self, args):
        super(SimpleModel, self).__init__()
        self.args = args
        self.device = args.device
        self.dr_rate = args.drop_out

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        self.embedding_tag = nn.Embedding(self.args.n_tag, self.hidden_dim)
        self.embedding_qclass = nn.Embedding(self.args.n_class, self.hidden_dim)
        self.embedding_question = nn.Embedding(self.args.n_questions, self.hidden_dim)
        
        self.batch_norm_tag = nn.BatchNorm1d(2)
        self.batch_norm_qclass = nn.BatchNorm1d(2)
        
        self.embedding_rate_tag = nn.utils.weight_norm(nn.Linear(2, self.hidden_dim))
        self.embedding_rate_qclass = nn.utils.weight_norm(nn.Linear(2, self.hidden_dim))
        
        self.layer_norm_tag = nn.LayerNorm(self.hidden_dim)
        self.layer_norm_qclass = nn.LayerNorm(self.hidden_dim)
        
        self.comb_proj = nn.utils.weight_norm(nn.Linear(self.hidden_dim*2, self.hidden_dim))
        self.layer_norm_knowledge = nn.LayerNorm(self.hidden_dim)
        
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        
        # final layer
        self.fc = nn.utils.weight_norm(nn.Linear(self.hidden_dim, 1))
        self.drop_out = nn.Dropout(self.dr_rate)
        
    def forward(self, input):
        (userID_KnowledgeTag_total_answer, userID_KnowledgeTag_acc,
         userID_question_class_total_answer, userID_question_class_acc,
         _, question, tag, qclass, _) = input
        
        batch_size = len(qclass)
        
        # user knowledge embedding
        
        ## tag embedding
        embed_tag = self.embedding_tag(tag)
        embed_tag = self.drop_out(embed_tag)
        
        embed_tag_rate = torch.cat([userID_KnowledgeTag_total_answer.unsqueeze(1),
                                    userID_KnowledgeTag_acc.unsqueeze(1)], 1)
        embed_tag_rate = self.batch_norm_tag(embed_tag_rate)
        embed_tag_rate = self.embedding_rate_tag(embed_tag_rate)
        embed_tag_rate = self.layer_norm_tag(embed_tag_rate)
        embed_tag_rate = self.prelu(embed_tag_rate)
        embed_tag = embed_tag * embed_tag_rate
        
        ## qclass embedding
        embed_qclass = self.embedding_qclass(qclass)
        embed_qclass = self.drop_out(embed_qclass)
        
        embed_qclass_rate = torch.cat([userID_question_class_total_answer.unsqueeze(1),
                                       userID_question_class_acc.unsqueeze(1)], 1)
        embed_qclass_rate = self.batch_norm_qclass(embed_qclass_rate)
        embed_qclass_rate = self.embedding_rate_qclass(embed_qclass_rate)
        embed_qclass_rate = self.layer_norm_qclass(embed_qclass_rate)
        embed_qclass_rate = self.prelu(embed_qclass_rate)
        embed_qclass = embed_qclass * embed_qclass_rate
        
        embed_knowledge = torch.cat([embed_tag,
                                     embed_qclass,
                                    ], 1)
        
        embed_knowledge = self.comb_proj(embed_knowledge)
        embed_knowledge = self.layer_norm_knowledge(embed_knowledge)
        embed_knowledge = self.prelu(embed_knowledge)
        embed_knowledge = self.drop_out(embed_knowledge)
        
        # mask - question embedding
        embed_question = self.embedding_question(question)
        embed_question = self.drop_out(embed_question)
        mask = self.sigmoid(embed_question)
        
        out = mask * embed_knowledge
        out = self.fc(out)
        preds = self.sigmoid(out).view(batch_size, -1)
        return preds
