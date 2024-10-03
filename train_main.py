import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import yaml
import random
import argparse
import datetime
from tqdm import tqdm
from agms_gcn import Model
from feeders.feeder import Feeder as Feeder

# CONFIG CLASS
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


#IMPORT CLASS
def import_class(name):    
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)      
    return mod

# SEED
def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# PROCESSOR CLASS
class Processor:

    def __init__(self, args):

        self.args = Config(args)    
    

        text = '!!!        MODEL PROPERTIES        !!!'
        line_length = 138
        padding = (line_length - len(text)) // 2
        centered_text = ' ' * padding + text + ' ' * padding
        print('\n' + '=' * line_length)
        print(centered_text)
        print('=' * line_length)
        print(f'-    Model Name       : {self.args.model_args.model_name}')
        print(f'-    Number of Classes: {self.args.model_args.num_classes}')
        print(f'-    Number of Persons: {self.args.model_args.num_person}')
        print(f'-    Number of Nodes  : {self.args.model_args.num_nodes}')
        print(f'-    Input channels   : {self.args.model_args.input_channels}')
        print(f'-    Graph            : {self.args.model_args.graph}')        
        print(f'-    Optimizer        : {self.args.optimizer_args.optimizer}')
        print(f'-    Base LR          : {self.args.optimizer_args.base_lr}')
        print(f'-    LR Step          : {self.args.optimizer_args.lr_step}')
        print(f'-    Weight Decay     : {self.args.optimizer_args.weight_decay}')
        print(f'-    Momentum         : {self.args.optimizer_args.momentum}')
        print(f'-    Nesterov         : {self.args.optimizer_args.nesterov}')
        print(f'-    Warm Up Epoch    : {self.args.optimizer_args.warm_up_epoch}')
        print(f'-    Dropout          : {self.args.model_args.dropout}')
        print(f'-    Batch Size       : {self.args.train_feeder_args.batch_size}')
        print(f'-    Number of Epochs : {self.args.num_epoch}')
        print(f'-    Train Data       : {self.args.train_feeder_args.data_path}')
        print(f'-    Train Label      : {self.args.train_feeder_args.label_path}')
        print(f'-    Test Data        : {self.args.test_feeder_args.data_path}')
        print(f'-    Test Label       : {self.args.test_feeder_args.label_path}')
        print('='*138 + '\n')

        if not os.path.isdir(self.args.work_dir):
            os.makedirs(self.args.work_dir)
            with open(self.args.work_dir + '/log.txt', 'w'):
                pass     
                
        # LOAD MODEL / OPTIMIZER        
        self.model = self.load_model()    
        self.optimizer = self.load_optimizer(self.model)      
        self.criterion = nn.CrossEntropyLoss()
             
        # LOAD TRAIN / TEST DATA
        self.data_loader = self.load_data()

        # LOSS / ACCURACY LIST
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []

        # CUDA AVAILABILITY
        if self.args.cuda:
            torch.backends.cudnn.benchmark = True

        self.start()

    # START    
    def start(self):
      
        text = '!!!       TRAIN   <<>>  EVALUATE     !!!'
        line_length = 138
        padding = (line_length - len(text)) // 2
        centered_text = ' ' * padding + text + ' ' * padding
        print('\n' + '=' * line_length)
        print(centered_text)
         
        print('='*138)
        print('{:>5} <-> {:>15} | {:>15} | {:>15}  <<>> {:>15} | {:>15} | {:>15} | {:>8}'.format(
            'Epoch', 'Train Mean Loss', 'GFE_Two_Accuracy', 'GFE_Three_Accuracy', 'Test Mean Loss', 'GFE_Two_Accuracy', 'GFE_Three_Accuracy', 'Time'))
        print('='*138)

        for epoch in range(1, self.args.num_epoch+1):

            self.adjust_learning_rate(epoch)

            mean_loss_train, acc_gfe_one_train, acc_gfe_two_train = self.train()
            mean_loss_eval, acc_gfe_one_eval, acc_gfe_two_eval = self.eval()

            self.train_loss_list.append(mean_loss_train)
            self.train_acc_list.append(acc_gfe_two_train)
            self.test_loss_list.append(mean_loss_eval)
            self.test_acc_list.append(acc_gfe_two_eval)

            
            self.print_log('{:>5} <-> {:>15.4f} | {:>16.4f} | {:>18.4f}  <<>> {:>15.4f} | {:>16.4f} | {:>18.4f} | {:>8}'.format(
                epoch, mean_loss_train, acc_gfe_one_train, acc_gfe_two_train, mean_loss_eval, acc_gfe_one_eval, acc_gfe_two_eval,datetime.datetime.now().strftime('%H:%M:%S')))


            # SAVE BEST MODEL
            if (len(self.test_acc_list) - 1) == np.argmax(self.test_acc_list):
                path = self.args.work_dir + '/best_model.pt'
                torch.save(self.model.state_dict(), path)

        self.print_log('- Best Test Accuracy: {:.3f}[%]'.format(np.max(self.test_acc_list)))

    # TRAIN 

    def train(self):

        loader = self.data_loader['train']
        process = tqdm(loader, dynamic_ncols=True,leave=False, desc='Train')

        loss_value = []
        acc_gfe_two= 0
        acc_gfe_three = 0

        for i, (data, label, _) in enumerate(process):

            with torch.set_grad_enabled(True):

                data = data.to(self.args.device)
                label = label.to(self.args.device)

                # output_gfe_two, output_gfe_three,_,_= self.model(data)
                output_gfe_two, output_gfe_three= self.model(data)

                loss = self.criterion(output_gfe_two, label) + self.criterion(output_gfe_three, label)                
                loss_value.append(loss.item())
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()             

                _, predict_label = torch.max(output_gfe_two.data, 1)
                acc_gfe_two += (predict_label == label).sum().item()

                _, predict_label = torch.max(output_gfe_three.data, 1)
                acc_gfe_three += (predict_label == label).sum().item()

        len_data_loader = len(self.data_loader['train'].dataset)
        return np.mean(loss_value), (100. * acc_gfe_two/len_data_loader), (100. * acc_gfe_three/len_data_loader)
    
    # def train(self): # gradient accumulation

    #     loader = self.data_loader['train']
    #     process = tqdm(loader, dynamic_ncols=True,leave=False, desc='Train')

    #     loss_value = []
    #     acc_gfe_two= 0
    #     acc_gfe_three = 0
    #     accumulation_steps = 8 

    #     for i, (data, label, _) in enumerate(process):

    #         with torch.set_grad_enabled(True):

    #             data = data.to(self.args.device)
    #             label = label.to(self.args.device)

    #             # output_gfe_two, output_gfe_three,_,_= self.model(data)
    #             output_gfe_two, output_gfe_three= self.model(data)

    #             loss = self.criterion(output_gfe_two, label) + self.criterion(output_gfe_three, label)                
    #             loss_value.append(loss.item())

    #             loss = loss / accumulation_steps
    #             loss.backward()
                
    #             if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
    #                 self.optimizer.step()
    #                 self.optimizer.zero_grad()                

    #             _, predict_label = torch.max(output_gfe_two.data, 1)
    #             acc_gfe_two += (predict_label == label).sum().item()

    #             _, predict_label = torch.max(output_gfe_three.data, 1)
    #             acc_gfe_three += (predict_label == label).sum().item()

    #     len_data_loader = len(self.data_loader['train'].dataset)
    #     return np.mean(loss_value), (100. * acc_gfe_two/len_data_loader), (100. * acc_gfe_three/len_data_loader)
    
    # EVALUATE
    def eval(self):

        loader = self.data_loader['test']
        process = tqdm(loader, dynamic_ncols=True,leave=False, desc='Test')
        loss_value = []
        acc_gfe_two = 0
        acc_gfe_three = 0


        with torch.no_grad():       
            for _, (data, label, _) in enumerate(process):
                
                data = data.to(self.args.device)
                label = label.to(self.args.device)
                
                # output_gfe_two, output_gfe_three,_,_= self.model(data)
                output_gfe_two, output_gfe_three= self.model(data)

                loss = self.criterion(output_gfe_two, label) + self.criterion(output_gfe_three, label)
                loss_value.append(loss.item())

                _, predict_label = torch.max(output_gfe_two.data, 1)
                acc_gfe_two += (predict_label == label).sum().item()

                _, predict_label = torch.max(output_gfe_three.data, 1)
                acc_gfe_three += (predict_label == label).sum().item()

        len_data_loader = len(self.data_loader['test'].dataset)
        return np.mean(loss_value), (100. * acc_gfe_two/len_data_loader), (100. * acc_gfe_three/len_data_loader)


    # def adjust_learning_rate(self, epoch): #v1
    #     if self.args.optimizer_args.optimizer== 'SGD' or self.args.optimizer_args.optimizer == 'Adam':
    #         if epoch < self.args.optimizer_args.warm_up_epoch:
    #             lr = self.args.optimizer_args.base_lr * (epoch + 1) / self.args.optimizer_args.warm_up_epoch
    #         else:
    #             lr = self.args.optimizer_args.base_lr * (0.5 * (np.cos((epoch-self.args.optimizer_args.warm_up_epoch) / (self.args.num_epoch-self.args.optimizer_args.warm_up_epoch) * np.pi) + 1))
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] = lr
    #         return lr
    #     else:
    #         raise ValueError()

    def adjust_learning_rate(self, epoch): #v2
        if self.args.optimizer_args.optimizer== 'SGD' or self.args.optimizer_args.optimizer == 'Adam':
            num_epoch_ = self.args.optimizer_args.cosine_epoch + self.args.optimizer_args.warm_up_epoch
            lr_cos = self.args.optimizer_args.base_lr * (0.5 * (np.cos((epoch-self.args.optimizer_args.warm_up_epoch) / (num_epoch_-self.args.optimizer_args.warm_up_epoch) * np.pi) + 1))
            if epoch < self.args.optimizer_args.warm_up_epoch:
                lr = self.args.optimizer_args.base_lr * (epoch + 1) / self.args.optimizer_args.warm_up_epoch  
            elif epoch < num_epoch_ and lr_cos > 0.01: 
                lr = lr_cos
            else:
                lr = self.args.optimizer_args.base_lr * (0.1 ** np.sum(epoch >= np.array(self.args.optimizer_args.lr_step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    # LOAD MODEL
    def load_model(self):

        if self.args.model_args.model_name is None:
            raise ValueError()
        else:
            model_class=import_class(self.args.model_args.model_name)


        model = model_class(self.args.model_args.num_classes,             
                            self.args.model_args.residual,
                            self.args.model_args.dropout,
                            self.args.model_args.num_person,                     
                            self.args.model_args.graph,      
                            self.args.model_args.num_nodes,   
                            self.args.model_args.input_channels        
                        )

        model = model.to(self.args.device)
        return model

    # LOAD OPTIMIZER
    def load_optimizer(self, model):
        if self.args.optimizer_args.optimizer == 'SGD':

            optimizer = optim.SGD(model.parameters(),
                                        lr=self.args.optimizer_args.base_lr,
                                        momentum=self.args.optimizer_args.momentum,
                                        nesterov=self.args.optimizer_args.nesterov,
                                        weight_decay=self.args.optimizer_args.weight_decay)
        elif self.args.optimizer_args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(),
                                        lr=self.args.optimizer_args.base_lr,
                                        weight_decay=self.args.optimizer_args.weight_decay)
        else:
            raise ValueError

        return optimizer

    # LOAD DATA
    def load_data(self):

        data_loader = dict()
        data_loader['train'] = torch.utils.data.DataLoader(dataset=Feeder(data_path=self.args.train_feeder_args.data_path,
                                                                        label_path=self.args.train_feeder_args.label_path,                                                                     
                                                                        normalization=self.args.train_feeder_args.normalization,
                                                                        random_shift=self.args.train_feeder_args.random_shift,                                          
                                                                        random_choose=self.args.train_feeder_args.random_choose,
                                                                        random_move=self.args.train_feeder_args.random_move,                                                              
                                                                       ),
                                                        batch_size=self.args.train_feeder_args.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.args.train_feeder_args.num_worker,
                                                        pin_memory=True)
        data_loader['test'] = torch.utils.data.DataLoader(dataset=Feeder(data_path=self.args.test_feeder_args.data_path,
                                                                        label_path=self.args.test_feeder_args.label_path,                                                                     
                                                                        normalization=self.args.test_feeder_args.normalization,
                                                                        random_shift=self.args.test_feeder_args.random_shift,                                                              
                                                                        random_choose=self.args.test_feeder_args.random_choose,
                                                                        random_move=self.args.test_feeder_args.random_move,                                                                   
                                                                       ),
                                                        batch_size=self.args.test_feeder_args.batch_size,
                                                        shuffle=False,
                                                        num_workers=0)

        return data_loader
    
    # PRINT LOG
    def print_log(self, input):
        print(input)
        with open(f'{self.args.work_dir}/log.txt', 'a') as f:
            f.write(input + '\n')

# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', type=str, default='config/ntu60_xview.yaml')
    p = parser.parse_args()
    init_seed() 
 
    with open(p.config, 'r') as f: 
        args = yaml.load(f, Loader=yaml.SafeLoader)
    Processor(args) 

