from torch.utils.data import Dataset, DataLoader
from model import _CNN
from model_class import _CNN_classification
from model_regr import _CNN_regression
from dataloader import CNN_Data
from loss import ConRegGroupLoss
from utils import matrix_sum, get_acc, get_MCC, get_confusion_matrix, write_raw_score, squared_error
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import ConcatDataset
from tqdm import tqdm
import wandb
import time


class CNN_Wrapper:
    def __init__(self,
                 fil_num,
                 drop_rate,
                 seed,
                 batch_size,
                 balanced,
                 data_dir,
                 data_dir_ex,
                 learn_rate,
                 train_epoch,
                 dataset,
                 external_dataset,
                 model_name,
                 metric,
                 device,
                 process):

        """
            :param fil_num:    channel number
            :param drop_rate:  dropout rate
            :param seed:       random seed
            :param batch_size: batch size for training CNN
            :param balanced:   balanced could take value 0 or 1, corresponding to different approaches to handle data
                               imbalance, see self.prepare_dataloader for more details
            :param model_name: give a name toed for saving model during training, can be either 'accuracy' or 'MCC' for
                               example, if metric == 'accuracy', then the time point where validation set has best
                               accuracy wi the model
            :param metric:     metric usll be saved
        """

        self.device = f'cuda:{device}'
        self.epoch = 0
        self.seed = seed
        self.Data_dir = data_dir
        self.Data_dir_ex = data_dir_ex
        self.learn_rate = learn_rate
        self.train_epoch = train_epoch
        self.balanced = balanced
        self.batch_size = batch_size
        self.dataset = dataset
        self.external_dataset = external_dataset
        self.model_name = model_name
        self.cross_index = None
        self.get_con_reg_group_loss = ConRegGroupLoss(self.device)
        self.eval_metric = get_acc if metric == 'accuracy' else get_MCC
        self.process = process
        if process == "valid":
            self.checkpoint_dir = './checkpoint_dir/valid/'
        else:
            self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, self.cross_index)
        self.model = _CNN(fil_num=fil_num, drop_rate=drop_rate).to(self.device)
        self.classification_model = _CNN_classification(fil_num=fil_num, drop_rate=drop_rate).to(self.device)
        self.regression_model = _CNN_regression(fil_num=fil_num, drop_rate=drop_rate).to(self.device)
        self.optimal_epoch = self.epoch
        self.optimal_valid_mse = 99999.0
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0.0
        self.frequency_dict = None

    def cross_validation(self, cross_index):
        self.cross_index = cross_index
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, self.cross_index)
        with open("lookupcsv/{}.csv".format(self.dataset), 'r') as csv_file:
            num = len(list(csv.reader(csv_file))) // 10
        start = int(self.cross_index * num)
        end = start + (num - 1)
        with open(self.checkpoint_dir + 'valid_result.txt', 'w') as file:
            file.write('')
        self.prepare_dataloader(start, end)

        if self.process == "train":
            self.train(distribution_loss=True)
        if self.process == "train_no_distr_loss":
            self.train(distribution_loss=False)
        if self.process == "test":
            self.test()
        if self.process == "classification":
            self.train_classification()
        if self.process == "regression":
            self.train_regression(distribution_loss=False)
        if self.process == "regression_distribution_loss":
            self.train_regression(distribution_loss=True)


    def validate(self):
        # Validate the model
        print('validating ...')

        # Load the model
        self.model.load_state_dict(torch.load('checkpoint_dir/valid/valid.pth', map_location='cuda:0'), strict=False)
        self.model.eval()
        with torch.no_grad():
            stage = "test"
            for i, dataset in enumerate(self.external_dataset):
                data_dir = self.Data_dir_ex[i]
                data = CNN_Data(data_dir, stage=stage, dataset=dataset, cross_index=self.cross_index, start=0,
                                end=-1, seed=self.seed)
                dataloader = DataLoader(data, batch_size=2, shuffle=False)
                f_clf = open(self.checkpoint_dir + 'raw_score_clf_{}_{}.txt'.format(dataset, stage), 'w')
                f_reg = open(self.checkpoint_dir + 'raw_score_reg_{}_{}.txt'.format(dataset, stage), 'w')
                matrix = [[0, 0], [0, 0]]
                mse = 0.0
                for idx, (inputs, labels, demors) in enumerate(dataloader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    clf_output, reg_output, _ = self.model(inputs)
                    write_raw_score(f_clf, clf_output, labels)
                    matrix = matrix_sum(matrix, get_confusion_matrix(clf_output, labels))
                    mse += squared_error(reg_output, demors)
                    write_raw_score(f_reg, reg_output, demors)
                mse /= (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
                print(dataset + "-" + stage + ' confusion matrix ', matrix)
                print('accuracy:', "%.4f" % self.eval_metric(matrix), 'and mean squared error ', mse)
                f_clf.close()
                f_reg.close()


    def prepare_dataloader(self, start, end):
        train_data = CNN_Data(self.Data_dir, stage='train', dataset=self.dataset, cross_index=self.cross_index,
                              start=start, end=end, seed=self.seed)
        valid_data = CNN_Data(self.Data_dir, stage='valid', dataset=self.dataset, cross_index=self.cross_index,
                              start=start, end=end, seed=self.seed)
        test_data = CNN_Data(self.Data_dir, stage='test', dataset=self.dataset, cross_index=self.cross_index,
                             start=start, end=end, seed=self.seed)
        self.frequency_dict = self.get_con_reg_group_loss.update(train_data.Label_list, train_data.demor_list)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()

        valid_data = ConcatDataset([valid_data, test_data])
        # the following if else blocks represent two ways of handling class imbalance issue
        if self.balanced == 1:
            # use pytorch sampler to sample data with probability according to the count of each class
            # so that each mini-batch has the same expectation counts of samples from each class
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif self.balanced == 0:
            # sample data from the same probability, but
            # self.imbalanced_ratio will be used in the weighted cross entropy loss to handle imbalanced issue
            self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                               num_workers=0)
        self.valid_dataloader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=0)


    def train(self, distribution_loss=True):
        # Train the model
        print("Fold {} is training ...".format(self.cross_index))

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))
        self.criterion_clf = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).to(self.device)
        self.criterion_reg = nn.SmoothL1Loss(reduction='mean').to(self.device)
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch = 0

        while self.epoch < self.train_epoch:
            self.train_model_epoch(distribution_loss)
            valid_matrix, valid_mse = self.valid_model_epoch()
            with open(self.checkpoint_dir + "valid_result.txt", 'a') as file:
                file.write(str(self.epoch) + ' ' + str(valid_matrix) + " " +
                           str(round(self.eval_metric(valid_matrix), 4)) + ' ' + str(valid_mse) + ' ' + '\n')
            print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix)
            print('eval_metric:', "%.4f" % self.eval_metric(valid_matrix), 'and mean squared error ', valid_mse.item())
            wandb.log({"val_mse": valid_mse, "val_acc": self.eval_metric(valid_matrix), "epoch": self.epoch})
            self.save_checkpoint(valid_matrix, valid_mse)
            self.epoch += 1
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric,
              self.optimal_valid_matrix)
        return self.optimal_valid_metric


    def train_model_epoch(self, distribution_loss=True):
        self.model.train()        
        for inputs, labels, demors in tqdm(self.train_dataloader, desc="Train Epoch"):
            # start_timer = time.perf_counter()
            inputs, labels, demors = inputs.to(self.device), labels.to(self.device), demors.to(self.device)
            self.model.zero_grad()
            # loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            # timer_inference = time.perf_counter()
            clf_output, reg_output, per_loss = self.model(inputs)
            # print(f"Time inference: {time.perf_counter() - timer_inference}")
            clf_loss = self.criterion_clf(clf_output, labels)
            reg_loss = self.criterion_reg(reg_output, torch.unsqueeze(demors, dim=1))
            # timer_loss = time.perf_counter()
            loss = clf_loss + reg_loss + torch.mean(per_loss)
            if distribution_loss:
                con_reg_group_loss = self.get_con_reg_group_loss.apply(reg_output, demors, self.frequency_dict, labels)  
                loss += torch.mean(con_reg_group_loss)
                wandb.log({"train_con_reg_group_loss": torch.mean(con_reg_group_loss), "epoch": self.epoch})
            # print(f"calculate loss time: {time.perf_counter() - timer_loss}") 
            wandb.log({"train_loss": loss, "train_clf_loss": clf_loss, "train_regr_loss": reg_loss, "train_interaction_loss": torch.mean(per_loss), "epoch": self.epoch})
            # timer_update = time.perf_counter()
            loss.backward()
            self.optimizer.step()
            # print(f"timer update: {time.perf_counter() - timer_update}")
            # print(f"train loop time: {time.perf_counter() - start_timer}")


    def valid_model_epoch(self):
        self.model.eval()
        with torch.no_grad():
            valid_matrix = [[0, 0], [0, 0]]
            mse = 0.0
            for inputs, labels, demors in tqdm(self.valid_dataloader, desc="Test Epoch"):
                inputs, labels, demors = inputs.to(self.device), labels.to(self.device), demors.to(self.device)
                clf_output, reg_output, _ = self.model(inputs)
                clf_loss = self.criterion_clf(clf_output, labels)
                reg_loss = self.criterion_reg(reg_output, torch.unsqueeze(demors, dim=1))
                loss = clf_loss + reg_loss
                wandb.log({"val_loss": loss, "val_clf_loss": clf_loss, "val_regr_loss": reg_loss, "epoch": self.epoch})
                valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(clf_output, labels))
                mse += squared_error(reg_output, demors)
            mse /= (valid_matrix[0][0] + valid_matrix[0][1] + valid_matrix[1][0] + valid_matrix[1][1])
        return valid_matrix, mse

    
    def train_classification(self):
        print("Fold {} is training ...".format(self.cross_index))

        self.optimizer = optim.Adam(self.classification_model.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))
        self.criterion_clf = nn.CrossEntropyLoss(weight=torch.Tensor([1, self.imbalanced_ratio])).to(self.device)
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch = 0

        while self.epoch < self.train_epoch:
            self.classification_model.train()            
            for inputs, labels, demors in tqdm(self.train_dataloader, desc="Train Epoch"):
                inputs, labels, demors = inputs.to(self.device), labels.to(self.device), demors.to(self.device)
                self.classification_model.zero_grad()
                clf_output = self.classification_model(inputs)
                clf_loss = self.criterion_clf(clf_output, labels)       
                loss = clf_loss
                wandb.log({"train_clf_loss": loss, "epoch": self.epoch})
                loss.backward()
                self.optimizer.step()
            
            self.classification_model.eval()
            with torch.no_grad():
                valid_matrix = [[0, 0], [0, 0]]
                for inputs, labels, demors in tqdm(self.valid_dataloader, desc="Test Epoch"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    clf_output = self.classification_model(inputs)
                    valid_matrix = matrix_sum(valid_matrix, get_confusion_matrix(clf_output, labels))
            valid_matrix = valid_matrix

            with open(self.checkpoint_dir + "classification_valid_result.txt", 'a') as file:
                file.write(str(self.epoch) + ' ' + str(valid_matrix) + " " +
                           str(round(self.eval_metric(valid_matrix), 4))+ '\n')
            print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix)
            print('eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
            wandb.log({"val_acc": self.eval_metric(valid_matrix), "epoch": self.epoch})
            # self.save_checkpoint(valid_matrix, valid_mse)
            self.epoch += 1
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric,
              self.optimal_valid_matrix)
        return self.optimal_valid_metric


    def train_regression(self, distribution_loss=False):
        print("Fold {} is training ...".format(self.cross_index))

        self.optimizer = optim.Adam(self.regression_model.parameters(), lr=self.learn_rate, betas=(0.5, 0.999))
        self.criterion_reg = nn.SmoothL1Loss(reduction='mean').to(self.device)
        self.optimal_valid_matrix = [[0, 0], [0, 0]]
        self.optimal_valid_metric = 0
        self.optimal_epoch = 0

        while self.epoch < self.train_epoch:
            self.regression_model.train()
            for inputs, labels, demors in tqdm(self.train_dataloader, desc="Train Epoch"):
                inputs, labels, demors = inputs.to(self.device), labels.to(self.device), demors.to(self.device)
                self.regression_model.zero_grad()
                reg_output = self.regression_model(inputs)
                reg_loss = self.criterion_reg(reg_output, torch.unsqueeze(demors, dim=1))
                loss = reg_loss
                if distribution_loss:
                    con_reg_group_loss = self.get_con_reg_group_loss.apply(reg_output, demors, self.frequency_dict, labels)            
                    loss += torch.mean(con_reg_group_loss)   
                    wandb.log({"train_con_reg_group_loss": torch.mean(con_reg_group_loss), "epoch": self.epoch})
                wandb.log({"train_loss": loss, "train_regr_loss": reg_loss, "epoch": self.epoch})
                loss.backward()
                self.optimizer.step()
            
            self.regression_model.eval()
            with torch.no_grad():
                mse = 0.0
                for inputs, labels, demors in tqdm(self.valid_dataloader, desc="Test Epoch"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    reg_output = self.regression_model(inputs)
                    mse += squared_error(reg_output, demors)
                mse /= len(self.valid_dataloader.dataset)
            
            wandb.log({"val_mse": mse, "epoch": self.epoch})
            out_file = "regression_valid_result.txt"
            if distribution_loss:
                out_file = f"dist_{out_file}"
            with open(self.checkpoint_dir + out_file, 'a') as file:
                file.write(str(self.epoch) + ' ' + str(mse) + '\n')
            print(f"{self.epoch}th epoch mean squared error: {mse}")
            self.epoch += 1

        return self.optimal_valid_metric
        

    def save_checkpoint(self, valid_matrix, valid_mse):
        # Choose the optimal model. The performance of AD detection task is prioritized
        if (self.eval_metric(valid_matrix) > self.optimal_valid_metric) or \
                (self.eval_metric(valid_matrix) == self.optimal_valid_metric and valid_mse < self.optimal_valid_mse):
            self.optimal_epoch = self.epoch
            self.optimal_valid_matrix = valid_matrix
            self.optimal_valid_metric = self.eval_metric(valid_matrix)
            self.optimal_valid_mse = valid_mse
            # for root, Dir, Files in os.walk(self.checkpoint_dir):
            #     for File in Files:
            #         if File.endswith("pth"):
            #             os.remove(os.path.join(self.checkpoint_dir, File))
            # torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name,
            #                                                          self.optimal_epoch))


    def test(self):
        # Test the model
        print('Fold {} is testing ... '.format(self.cross_index))

        # Load the optimal model
        if self.epoch == 0:
            for root, dirs, files in os.walk(self.checkpoint_dir):
                for file in files:
                    if file[-4:] == '.pth':
                        info = file[4:-4].split('_')
                        self.optimal_epoch = int(info[0])
        self.model.load_state_dict(torch.load('{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name,
                                                                   self.optimal_epoch)), strict=False)
        self.model.eval()
        with torch.no_grad():
            for dataset in [self.dataset]:# + self.external_dataset:
                for stage in ['train', 'valid', 'test']:
                    data_dir = self.Data_dir
                    if dataset != self.dataset:
                        if stage != 'test':
                            continue
                        data_dir = data_dir.replace('ADNI1', dataset)
                        data = CNN_Data(data_dir, stage=stage, dataset=dataset, cross_index=self.cross_index, start=0,
                                        end=-1, seed=self.seed)
                        dataloader = DataLoader(data, batch_size=2, shuffle=False)
                    elif stage == 'train':
                        dataloader = self.train_dataloader
                    elif stage == 'valid':
                        dataloader = self.valid_dataloader
                    else:
                        dataloader = self.test_dataloader
                    f_clf = open(self.checkpoint_dir + 'raw_score_clf_{}_{}.txt'.format(dataset, stage), 'w')
                    f_reg = open(self.checkpoint_dir + 'raw_score_reg_{}_{}.txt'.format(dataset, stage), 'w')
                    matrix = [[0, 0], [0, 0]]
                    mse = 0.0
                    for idx, (inputs, labels, demors) in enumerate(dataloader):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        clf_output, reg_output, _ = self.model(inputs)
                        write_raw_score(f_clf, clf_output, labels)
                        matrix = matrix_sum(matrix, get_confusion_matrix(clf_output, labels))
                        mse += squared_error(reg_output, demors)
                        write_raw_score(f_reg, reg_output, demors)
                    mse /= (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
                    print(dataset + "-" + stage + ' confusion matrix ', matrix)
                    print('accuracy:', "%.4f" % self.eval_metric(matrix), 'and mean squared error ', mse)
                    f_clf.close()
                    f_reg.close()
