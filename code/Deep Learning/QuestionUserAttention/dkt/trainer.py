import os
import torch
import numpy as np
import pandas as pd


from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .my_model import Mymodel

import wandb

def run(args, train_data, valid_data, fold):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    # 총 10번 warmup
    args.warmup_steps = args.total_steps // 10
            
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)
        
        ### VALID
        auc, acc,_ , _ = validate(valid_loader, model, args)


        ### TODO: model save or early stopping
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                  "valid_auc":auc, "valid_acc":acc})
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                },
                args.model_dir, f'model{fold}.pt',
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            #print(optimizer)
            scheduler.step()

    return best_auc


def train(train_loader, model, optimizer, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)
        preds = model(input)
        targets = input[3] # correct


        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
        
        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
        
        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)
      

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc}')
    return auc, acc, loss_avg
    

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[3] # correct


        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]
    
        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    
    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets



def inference(args, test_data):
    
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)
    
    
    total_preds = []
    
    for step, batch in enumerate(test_loader):
        input = process_batch(batch, args)

        preds = model(input)
        

        # predictions
        preds = preds[:,-1]
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)
        
    print(total_preds)
    write_path = os.path.join(args.output_dir, f"output_{args.fold}.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))


def staking_inference(args, test_data, fold, df):
    
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)
    
    validate(test_loader, model, args)
    
    total_preds = []
    
    for step, batch in enumerate(test_loader):
        input = process_batch(batch, args)

        preds = model(input)
        

        # predictions
        preds = preds[:,-1]
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)
    
    df["pred"] = total_preds
    df.to_csv(f"/opt/ml/code/output/output_{fold}.csv", index=False)
    # write_path = os.path.join(args.output_dir, f"output_{fold}.csv")
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)    
    # with open(write_path, 'w', encoding='utf8') as w:
    #     print("writing prediction : {}".format(write_path))
    #     w.write("id,prediction\n")
    #     for id, p in enumerate(total_preds):
    #         w.write('{},{}\n'.format(id,p))


def kfold_inference(args, test_data):
    kfold_ensemble = []
    model_name = args.model_name.split(".")[0] 
    for i in range(5):
        args.model_name = "model" + f"{i}.pt"
        
        model = load_model(args)
        model.eval()
        _, test_loader = get_loaders(args, None, test_data)
        
        
        total_preds = []
        
        for step, batch in enumerate(test_loader):
            input = process_batch(batch, args)

            preds = model(input)
            

            # predictions
            preds = preds[:,-1]

            if args.device == 'cuda':
                preds = preds.to('cpu').detach().numpy()
            else: # cpu
                preds = preds.detach().numpy()
                
            total_preds+=list(preds)
        #print(total_preds)
        kfold_ensemble.append(total_preds)

    answer = []
    for i in range(len(total_preds)):
        ensemble = 0
        for j in range(5):
            ensemble += kfold_ensemble[j][i]
        answer.append(ensemble / 5)

    write_path = os.path.join(args.output_dir, "output.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(answer):
            w.write('{},{}\n'.format(id,p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'lstmattn': model = LSTMATTN(args)
    if args.model == 'bert': model = Bert(args)
    if args.model == 'QUA': model = Mymodel(args)
    print(args.model)
    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):

    # test, question, tag, correct, mask = batch
    test, question, tag, correct, userID_elapsed_cate, IK_question_acc, question_class, IK_KnowledgeTag_acc, userID_acc, userID_assessmentItemID_experience, user_question_class_solved, solved_question, userID_KnowledgeTag_total_answer, userID_KnowledgeTag_acc, userID_acc_rolling, mask = batch
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다

    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction[:, 0] = 0 # set padding index to the first sequence
    interaction = (interaction * mask).to(torch.int64)
    # print(interaction)
    # exit()

    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)
    # 추가 feature
    userID_elapsed_cate = ((userID_elapsed_cate + 1) * mask).to(torch.int64)
    IK_question_acc = ((IK_question_acc + 1) * mask).to(torch.float32)
    question_class = ((question_class + 1) * mask).to(torch.int64)
    IK_KnowledgeTag_acc = ((IK_KnowledgeTag_acc + 1) * mask).to(torch.float32)
    userID_acc = ((userID_acc + 1) * mask).to(torch.float32)
    userID_assessmentItemID_experience = ((userID_assessmentItemID_experience + 1) * mask).to(torch.int64)
    user_question_class_solved = ((user_question_class_solved + 1) * mask).to(torch.float32)
    solved_question = ((solved_question + 1) * mask).to(torch.float32)
    userID_KnowledgeTag_total_answer = ((userID_KnowledgeTag_total_answer + 1) * mask).to(torch.float32)
    userID_KnowledgeTag_acc = ((userID_KnowledgeTag_acc + 1) * mask).to(torch.float32)
    userID_acc_rolling = ((userID_acc_rolling + 1) * mask).to(torch.float32)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1


    # device memory로 이동

    test = test.to(args.device)
    question = question.to(args.device)


    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)

    #feature 추가
    userID_elapsed_cate = userID_elapsed_cate.to(args.device)
    IK_question_acc = IK_question_acc.to(args.device)
    question_class = question_class.to(args.device)
    IK_KnowledgeTag_acc = IK_KnowledgeTag_acc.to(args.device)
    userID_acc = userID_acc.to(args.device)
    userID_assessmentItemID_experience = userID_assessmentItemID_experience.to(args.device)
    user_question_class_solved = user_question_class_solved.to(args.device)
    solved_question = solved_question.to(args.device)
    userID_KnowledgeTag_total_answer = userID_KnowledgeTag_total_answer.to(args.device)
    userID_KnowledgeTag_acc = userID_KnowledgeTag_acc.to(args.device)
    userID_acc_rolling = userID_acc_rolling.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    # return (test, question,
    #         tag, correct, mask,
    #         interaction, gather_index)

    return (test, question, tag, correct, userID_elapsed_cate, IK_question_acc, question_class, IK_KnowledgeTag_acc, userID_acc, userID_assessmentItemID_experience, user_question_class_solved, solved_question, userID_KnowledgeTag_total_answer, userID_KnowledgeTag_acc, userID_acc_rolling, mask, interaction, gather_index)


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
    #마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:,-1]
    loss = torch.mean(loss)
    return loss

def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()



def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))



def load_model(args):
    
    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
   
    
    print("Loading Model from:", model_path, "...Finished.")
    return model