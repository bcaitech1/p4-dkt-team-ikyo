import os
import torch
import numpy as np

from simple.dkt.dataloader import get_loaders
from simple.dkt.optimizer import get_optimizer
from simple.dkt.scheduler import get_scheduler
from simple.dkt.criterion import *
from simple.dkt.metric import get_metric
from simple.dkt.model import *

import wandb

def run(args, train_data, valid_data):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10
    
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    pre_targets = []
    pre_preds = []
    gamma = 0.2
    
    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        train_auc, train_acc, train_loss, pre_preds, pre_targets, gamma = train(train_loader, model, optimizer, args, epoch, pre_targets, pre_preds, gamma)
        
        ### VALID
        auc, acc, _ , _ = validate(valid_loader, model, args)
        
        ### TODO: model save or early stopping
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc, "valid_auc":auc, "valid_acc":acc})
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                },
                args.model_dir, 'model.pt',
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
            scheduler.step()
        break

def train(train_loader, model, optimizer, args, epoch, pre_targets, pre_preds, gamma):
    model.train()
    
    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)
        preds = model(input)
        targets = input[-5].view(-1, 1) # correct

        if args.loss == 'bce':
            loss = compute_loss(preds, targets)
        elif args.loss == 'auroc':
            if epoch > 0:
                loss = roc_star_loss(targets, preds, gamma, pre_targets, pre_preds)
            else:
                loss = compute_loss(preds, targets)
                pre_preds = preds
                pre_targets = targets
            
            gamma = epoch_update_gamma(targets, preds, epoch, 2)
            
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
    return auc, acc, loss_avg, pre_preds, pre_targets, gamma
    

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[-5].view(-1, 1) # correct

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
    
    total_userid = []
    total_preds = []
    
    for step, batch in enumerate(test_loader):
        input = process_batch(batch, args)

        preds = model(input)
        userid = input[-1]
        
        # predictions
        preds = preds[:,-1]
        
        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()

        total_userid += list(userid)
        total_preds += list(preds)
    
    write_path = os.path.join(args.output_dir, args.output_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,userid,prediction\n")
        for id, (u, p) in enumerate(zip(total_userid, total_preds)):
            w.write(f'{id},{u},{p}\n')

def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'simple': model = SimpleModel(args)
    else:
        raise NameError('model name error') 
    
    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):
    precessing = []
    for i, b in enumerate(batch): 
        if len(batch)-i == 5:
            b = b.type(torch.FloatTensor).to(args.device)
        elif len(batch)-i == 1:
            b = b
        elif len(batch)-i < 5:
            b = b.to(torch.int64).to(args.device)
        else:
            b = b.to(torch.float32).to(args.device)
        precessing.append(b)
    return precessing


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
#     loss = get_criterion(preds[:,-1:], targets[:,-1:])
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
    
    model_path = os.path.join(args.model_dir, args.load_model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
     
    print("Loading Model from:", model_path, "...Finished.")
    return model