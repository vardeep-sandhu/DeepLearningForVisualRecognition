import numpy as np
import torch
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        labels = torch.tensor(dict.get(b'labels'), dtype=torch.int)
        data = torch.tensor(dict.get(b'data'))
    return (labels, data)


def iter_files(path):
    train_list = []
    test_list = []
    for filename in sorted(os.listdir(path)):
        if filename == "test_batch":
            filepath = os.path.join(path, filename)
            stack_ = unpickle(filepath)
            test_list.append(stack_)
        else:
            filepath = os.path.join(path, filename)
            stack_ = unpickle(filepath)
            train_list.append(stack_)      
    return train_list, test_list

def batch2images(batch):
    batch_ = torch.zeros(len(batch), 32,32,3)
    for i in range(len(batch)):
        batch_[i]  = flatten2img(batch[i])
    return batch_

def flatten2img(flatten: torch.Tensor):
    img_R = flatten[0:1024].reshape((32, 32))
    img_G = flatten[1024:2048].reshape((32, 32))
    img_B = flatten[2048:3072].reshape((32, 32))
    img = torch.dstack((img_R, img_G, img_B))
    return img


# def train(model, dataloader, optimizer, reg: bool, EPOCHS):
#     losses_list = []
#     for epoch in range(EPOCHS):
#         running_loss = 0
#         for x,y in dataloader:
#             model.zero_grad()
#             prediction = model.forward(x)
#             norm = 0
#             if reg == True:
#                 for p in model.parameters():
#                     norm += torch.norm(p, 'fro')
#             loss = nn.CrossEntropyLoss()
#             output_loss = loss(prediction, y.to(torch.long)) + 0.001*(norm*norm)
#             output_loss.backward()
#             optimizer.step()
#             running_loss += output_loss.item()
#         losses_list.append(running_loss/len(dataloader))
#         print('Epoch: {}, Loss: {}'.format(epoch+1, running_loss/len(dataloader)))    
#     return losses_list

# def train(model, dataloader, optimizer, reg:bool):
#     for x,y in dataloader:
#         x, y = x.to('cuda:0'), y.to('cuda:0')
#         optimizer.zero_grad()
#         prediction = model.forward(x)
#         norm = 0
#         if reg == True:
#             for p in model.parameters():
#                 norm += torch.norm(p, 'fro')
#         loss = nn.CrossEntropyLoss()
#         output_loss = loss(prediction, y.to(torch.long)) + 0.001*(norm*norm)
#         output_loss.backward()
#         optimizer.step()
#     return output_loss
        
# def accuracy(model, dataloader):
#     corect = 0
#     total = 0 
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.to('cuda:0'), y.to('cuda:0')
#             prediction = model.forward(x)
#             for idx, i in enumerate(prediction):

#                 if torch.argmax(i) == y[idx]:
#                     corect += 1
#                 total += 1
#     acc = round(corect/total,3)
#     return acc        

# #testing 
# def accuracy(model, test_loader):
#     corect = 0
#     total = 0 
#     with torch.no_grad():
#         for x, y in test_loader:
#             prediction = model.forward(x)
#             for idx, i in enumerate(prediction):

#                 if torch.argmax(i) == y[idx]:
#                     corect += 1
#                 total += 1
#     acc = round(corect/total,3)
#     return acc


# from tqdm import tqdm
# from time import sleep

# def cross_validation(num_folds: int, dataset, model_factory, optimizer_factory, epochs):
#     fold_size = len(dataset) // num_folds
#     splits = torch.utils.data.random_split(dataset, (fold_size,) *num_folds)
#     accuracies = torch.zeros((num_folds, epochs))
#     losses = torch.zeros((num_folds, epochs))
    
#     for i in tqdm(range(num_folds)):
#         train_dataset = torch.utils.data.ConcatDataset([*splits[:i], *splits[i+1:]])
#         train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= 64, shuffle=True)
#         test_dataloader = torch.utils.data.DataLoader(splits[i], batch_size=1_024)
        
#         model = model_factory.to('cuda:0')
# #         model = model_factory
#         optimizer = optimizer_factory
        
#         for epoch in tqdm(range(epochs)):
# #             print(f"Losses and Accuracy of Cross Val: {i+1}")
#             losses[i, epoch] = train(model, train_dataloader, optimizer, True)
#             accuracies[i, epoch] = acc(model, test_dataloader)
# #             print(f"{epoch+1}{',' if epoch != epochs-1 else ''}", end="")
#             sleep(0.1)
#         sleep(0.01)
# #         print()
#     return accuracies, losses

# reg_acc, reg_losses = cross_validation(5, train_dataset, model1, optimizer, 25)