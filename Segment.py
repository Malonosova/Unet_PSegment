import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchmetrics import JaccardIndex
from tqdm import tqdm
import torch
import numpy as np

class Segmentation:
    def __init__(self, model, loss, optimizer, epochs=100, device='cuda', save_path="./save_model/"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss.to(self.device)
        self.opt = optimizer
        self.epochs = epochs
        self.save_path = save_path
        
    @staticmethod   
    def predict(model, img, device='cuda'):
        model.eval()
        return model(img.unsqueeze(0))
    
    @staticmethod
    def split(y_batch):
        y_batch_n = lambda x: (y_batch==x)
        y_batch_body = (y_batch>0).long()
                
        y_batch_up = (y_batch_n(1)+y_batch_n(6)+y_batch_n(2)+y_batch_n(4)>0)*2
        y_batch_low = torch.logical_or(y_batch_n(3) , y_batch_n(5))*1
        y_batch_half = (y_batch_up+y_batch_low).long()
        return y_batch_body, y_batch_half
    
    @staticmethod
    def metric(pred, target, num_classes, device='cuda'):
        jaccard = lambda x: JaccardIndex(task='multiclass', num_classes=x).to(device)
        iou = jaccard(num_classes)
        return iou(pred.argmax(dim=1), target)
    
    def save_txt(self, history):
        with open('out_lr1_.txt', 'w') as f:
            f.write('epoch\ttrain_loss\ttrain_iou\tval_iou\n')
            for ind, i in enumerate(np.array(history).T):
                f.write(str(ind)+'\t')
                for k in i:
                    f.write(str(k)+'\t')
                f.write('\n')
            f.close
    
    def Plot_train(self, loss, val_loss, iou_train, iou_val):
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(np.array([loss, val_loss]).T)
        plt.legend(['train loss', 'val loss'])
        plt.title('loss')

        plt.subplot(1, 3, 2)
        plt.plot(iou_train)
        plt.legend(['body', 'half body', 'each part'])
        plt.title('train IoU')
        
        plt.subplot(1, 3, 3)
        plt.plot(iou_val)
        plt.legend(['body', 'half body', 'each part'])
        plt.title('val IoU')
        plt.show()
        
    def evaluation(self, model, val_loader):
        model.eval()
        avg_val_loss, avg_val_iou = 0, []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).long().squeeze()
                y_batch_body, y_batch_half = self.split(y_batch)

                out0, out1, out2 = model(x_batch)
                loss = self.loss_fn(out0,y_batch_body)+2*self.loss_fn(out1,y_batch_half)+5*self.loss_fn(out2,y_batch)
                
                avg_iou0 = self.metric(out0, y_batch_body, 2)
                avg_iou1 = self.metric(out1, y_batch_half, 3)
                avg_iou2 = self.metric(out2, y_batch, 7)
                
                avg_val_iou.append([avg_iou0.cpu(), avg_iou1.cpu(), avg_iou2.cpu()])
                avg_val_loss += loss
                
            avg_val_loss /= len(val_loader)
        return avg_val_iou, avg_val_loss
        
    def fit(self, train_loader, val_loader):
        train_loss, val_loss, train_iou, val_iou = [], [], [], []
        
        for i in tqdm(range(self.epochs)):
            avg_train_loss, avg_train_iou = 0, []
            
            
            self.model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).long().squeeze()
                y_batch_body, y_batch_half = self.split(y_batch)
                
                self.opt.zero_grad()
                out0, out1, out2 = self.model(x_batch)
            
                loss = self.loss_fn(out0,y_batch_body)+2*self.loss_fn(out1,y_batch_half)+5*self.loss_fn(out2,y_batch)
                loss.backward()
                self.opt.step()
                
                avg_iou0 = self.metric(out0, y_batch_body, 2)
                avg_iou1 = self.metric(out1, y_batch_half, 3)
                avg_iou2 = self.metric(out2, y_batch, 7)
                
                avg_train_loss += loss
                avg_train_iou.append([avg_iou0.cpu(), avg_iou1.cpu(), avg_iou2.cpu()])
                
            avg_train_loss /= len(train_loader)
            avg_val_iou, avg_val_loss = self.evaluation(self.model, val_loader)
            
            train_loss.append(round(avg_train_loss.item(), 4))
            val_loss.append(round(avg_val_loss.item(), 4))
            train_iou.append(np.mean(avg_train_iou, axis=0))
            val_iou.append(np.mean(avg_val_iou, axis=0))
            
            if i%10==0:
                clear_output(wait=True)
                self.Plot_train(train_loss, val_loss, train_iou, val_iou)
                torch.save(self.model.state_dict(), self.save_path+f'model_lr1_epochs_{i}')
        self.save_txt([train_loss, train_iou, val_iou])
        return train_loss, train_iou, val_iou