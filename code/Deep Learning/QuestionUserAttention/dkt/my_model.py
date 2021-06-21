import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel    

class Feed_Forward_block(nn.Module):
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff , out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff , out_features=dim_ff)

    def forward(self,ffn_in):
        return  self.layer2(   F.relu( self.layer1(ffn_in) )   )


class Mymodel(nn.Module):

    def __init__(self, args):
        super(Mymodel, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        # n_test, n_questions, n_tag : 유니크한 개수??
        
        self.embed_dim = 128

        self.embedding_interaction = nn.Embedding(3, self.embed_dim)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.embed_dim)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.embed_dim)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.embed_dim)

        # 추가 feature
        self.embedding_userID_elapsed_cate = nn.Embedding(self.args.n_userID_elapsed_cate + 1, self.embed_dim)
        self.embedding_question_class = nn.Embedding(self.args.n_question_class + 1, self.embed_dim)
        self.embedding_userID_assessmentItemID_experience = nn.Embedding(self.args.n_userID_assessmentItemID_experience + 1, self.embed_dim)

        self.drop_out = nn.Dropout(self.args.drop_out)

        # embedding combination projection
        self.question_cate= nn.Linear(self.embed_dim*5, self.hidden_dim // 2)
        self.question_cont = nn.Linear(2, self.hidden_dim // 2)
        self.user_cate= nn.Linear(self.embed_dim*2, self.hidden_dim // 2)
        self.user_cont = nn.Linear(5, self.hidden_dim // 2)
        self.prelu1_question = nn.PReLU()
        self.prelu1_user = nn.PReLU()

        self.attention_layer = nn.MultiheadAttention(embed_dim= self.hidden_dim, num_heads= 8, dropout=self.args.drop_out)
        self.ff_layer = Feed_Forward_block(self.hidden_dim)
        self.prelu2_attention = nn.PReLU()


        self.question_layer_norm = nn.LayerNorm(self.hidden_dim)
        self.user_layer_norm = nn.LayerNorm(self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.conv1d_layer1 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.prelu3_conv1 = nn.PReLU()
        self.conv1d_layer2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=2)
        self.prelu3_conv2 = nn.PReLU()
        self.conv1d_layer3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3)
        self.prelu3_conv3 = nn.PReLU()
        self.conv1d_layer4 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=4)
        self.prelu3_conv4 = nn.PReLU()
        self.conv1d_layer5 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5)
        self.prelu3_conv5 = nn.PReLU()
        self.conv1d_layer6 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=6)
        self.prelu3_conv6 = nn.PReLU()
        self.conv1d_layer7 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=7)
        self.prelu3_conv7 = nn.PReLU()
        self.conv1d_layer8 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=8)
        self.prelu3_conv8 = nn.PReLU()
        self.conv1d_layer9 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9)
        self.prelu3_conv9 = nn.PReLU()
        self.conv1d_layer10 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=10)
        self.prelu3_conv10 = nn.PReLU()


        self.lstm = nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers=1)
        self.fc = nn.Linear(self.hidden_dim , 1)
        self.activation = nn.Sigmoid()


    def forward(self, input):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        #test, question, tag, _, mask, interaction, _ = input
        test, question, tag, _, userID_elapsed_cate, IK_question_acc, question_class, IK_KnowledgeTag_acc, userID_acc, userID_assessmentItemID_experience, user_question_class_solved, solved_question, userID_KnowledgeTag_total_answer, userID_KnowledgeTag_acc, userID_acc_rolling, mask, interaction, _ = input
        batch_size = interaction.size(0)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)        

        
        embed_userID_elapsed_cate = self.embedding_userID_elapsed_cate(userID_elapsed_cate)
        embed_question_class = self.embedding_question_class(question_class)
        embed_userID_assessmentItemID_experience = self.embedding_userID_assessmentItemID_experience(userID_assessmentItemID_experience)

        question_cate = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,
                           embed_question_class
                           ], 2)

        question_cont = torch.cat([IK_question_acc.unsqueeze(-1),
                           IK_KnowledgeTag_acc.unsqueeze(-1),
                           ], 2)

        user_cate = torch.cat([
                           embed_userID_elapsed_cate,
                           embed_userID_assessmentItemID_experience,
                           ], 2)

        user_cont = torch.cat([
                           userID_acc.unsqueeze(-1),
                           user_question_class_solved.unsqueeze(-1),
                           solved_question.unsqueeze(-1),
                           userID_KnowledgeTag_total_answer.unsqueeze(-1),
                           userID_KnowledgeTag_acc.unsqueeze(-1),
                           ], 2)

        question_cate_embed = self.question_cate(question_cate)
        question_cont_embed = self.question_cont(question_cont)

        user_cate_embed = self.user_cate(user_cate)
        user_cont_embed = self.user_cont(user_cont)

        question_embed = self.drop_out((torch.cat([question_cate_embed, question_cont_embed], 2)))
        user_embed = self.drop_out((torch.cat([user_cate_embed, user_cont_embed], 2)))

        question_embed = self.question_layer_norm(question_embed)
        user_embed = self.question_layer_norm(user_embed)

        question_embed = self.prelu1_question(question_embed)
        user_embed = self.prelu1_user(user_embed)

        out, attn_wt = self.attention_layer(question_embed , user_embed , user_embed)
        out = self.ff_layer(out)
        out = self.prelu2_attention(out)

        out = out + user_embed + question_embed

        conv_output1 = self.prelu3_conv1(self.conv1d_layer1(out.transpose(1, 2))) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output2 = self.prelu3_conv2(self.conv1d_layer2(out.transpose(1, 2))) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output3 = self.prelu3_conv3(self.conv1d_layer3(out.transpose(1, 2))) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output4 = self.prelu3_conv4(self.conv1d_layer4(out.transpose(1, 2))) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output5 = self.prelu3_conv5(self.conv1d_layer5(out.transpose(1, 2))) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output6 = self.prelu3_conv6(self.conv1d_layer6(out.transpose(1, 2))) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output7 = self.prelu3_conv7(self.conv1d_layer7(out.transpose(1, 2))) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output8 = self.prelu3_conv8(self.conv1d_layer8(out.transpose(1, 2))) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output9 = self.prelu3_conv9(self.conv1d_layer9(out.transpose(1, 2))) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output10 = self.prelu3_conv10(self.conv1d_layer10(out.transpose(1, 2))) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)

        out = conv_output1[:,:,9:] + conv_output2[:,:,8:] + conv_output3[:,:,7:] + conv_output4[:,:,6:] + conv_output5[:,:,5:] + conv_output6[:,:,4:] + conv_output7[:,:,3:] + conv_output8[:,:,2:] + conv_output9[:,:,1:] + conv_output10
        out = out.transpose(1, 2)
        out = out + user_embed[:,9:,:] + question_embed[:,9:,:]

        #out = self.drop_out(out)
        #out, _ = self.lstm(out)?

        out = self.drop_out(out)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds