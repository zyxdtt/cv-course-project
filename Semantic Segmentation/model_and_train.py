import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
from torchvision import transforms,models
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

img_height=256
img_width=1600
num_classes=4
learning_rate=1e-4
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform=transforms.Compose([
    transforms.Resize((img_height,img_width),interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

val_transform=transforms.Compose([
    transforms.Resize((img_height,img_width),interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

def rle_decode(rle_str,shape=(img_height,img_width)):
    if pd.isna(rle_str) or rle_str=='1 409600':
        return np.zeros(shape,dtype=np.uint8)
    s=list(map(int,rle_str.split()))
    mask=np.zeros(shape[0]*shape[1],dtype=np.uint8)
    for start,length in zip(s[0::2],s[1::2]):
        mask[start-1:start-1+length]=1
    return mask.reshape(shape,order='F')

def rle_encoder(mask):
    pixels=mask.T.flatten()
    if np.sum(pixels)==0:
        return '1 409600'
    pixels=np.concatenate([[0],pixels,[0]])
    runs=np.where(pixels[1:]!=pixels[:-1])[0]+1
    runs[1::2]-=runs[0::2]
    return ' '.join(str(x) for x in runs)

class SteelDataset(Dataset):
    def __init__(self,df,img_dir,transform=None):
        self.df=df
        self.img_dir=img_dir
        self.transform=transform
        self.image_ids=df['ImageId'].unique()

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        img_id=self.image_ids[index]
        img_path=os.path.join(self.img_dir,img_id)
        image=Image.open(img_path).convert('RGB')
        if self.transform:
            image=self.transform(image)
        masks=[]
        for class_id in range(1,num_classes+1):
            row=self.df[(self.df['ImageId']==img_id)&(self.df['ClassId']==class_id)]
            if len(row)>0:
                rle=row['EncodedPixels'].values[0]
                mask=rle_decode(rle)
            else:
                mask=np.zeros((img_height,img_width),dtype=np.uint8)
            masks.append(mask)
        mask=np.stack(masks,axis=0)
        mask=torch.from_numpy(mask).float()
        return image,mask,img_id

class TestDataset(Dataset):
    def __init__(self,img_dir,transform=None):
        self.img_dir=img_dir
        self.img_ids=[f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform=transform if transform else transforms.Compose([
            transforms.Resize((img_height,img_width),interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        img_id=self.img_ids[index]
        img_path=os.path.join(self.img_dir,img_id)
        image=Image.open(img_path).convert('RGB')
        if self.transform:
            image=self.transform(image)
        return image,img_id

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.Conv=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.Conv(x)
    
class Up(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2)
        self.conv=DoubleConv(out_ch*2,out_ch)
    
    def forward(self,x1,x2):
        x1=self.up(x1)
        diffy=x2.size()[2]-x1.size()[2]
        diffx=x2.size()[3]-x1.size()[3]
        x1=F.pad(x1,[diffx//2,diffx-diffx//2,diffy//2,diffy-diffy//2])
        x=torch.cat([x2,x1],dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,n_classes=4):
        super().__init__()
        resnet=models.resnet34(pretrained=True)
        self.enc1=nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu)#64
        self.enc2=nn.Sequential(resnet.maxpool,resnet.layer1)#64
        self.enc3=resnet.layer2#128
        self.enc4=resnet.layer3#256
        self.enc5=resnet.layer4#512
        self.center=DoubleConv(512,512)
        self.up1=Up(512,256)
        self.up2=Up(256,128)
        self.up3=Up(128,64)
        self.up4=Up(64,64)
        self.final_up=nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.out=nn.Conv2d(64,n_classes,kernel_size=1)
    
    def forward(self,x):
        e1=self.enc1(x)#1/2,64
        e2=self.enc2(e1)#1/4,64
        e3=self.enc3(e2)#1/8,128
        e4=self.enc4(e3)#1/16,256
        e5=self.enc5(e4)#1/32,512
        center=self.center(e5)
        d1=self.up1(center,e4)#256
        d2=self.up2(d1,e3)#128
        d3=self.up3(d2,e2)#64
        d4=self.up4(d3,e1)#64
        d5=self.final_up(d4)
        out=torch.sigmoid(self.out(d5))
        return out

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce=nn.BCELoss()
    def forward(self,pred,tar):
        bce=self.bce(pred,tar)
        pred_flat=pred.view(pred.size(0),-1)
        tar_flat=tar.view(tar.size(0),-1)
        intersaction=(pred_flat*tar_flat).sum(dim=1)
        dice=(2.*intersaction+1e-6)/(pred_flat.sum(dim=1)+tar_flat.sum(dim=1)+1e-6)
        dice=dice.mean()
        return bce+1-dice
    
def train(model,train_loader,val_loader,epochs=10):
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    sche=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=2)
    loss=Loss()
    best_val_loss=float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss=0
        for images,masks,_ in tqdm(train_loader,desc=f'Epoch {epoch+1}/{epochs} Train'):
            images,masks=images.to(device),masks.to(device)
            opt.zero_grad()
            pred=model(images)
            l=loss(pred,masks)
            l.backward()
            opt.step()
            train_loss+=l.item()
        train_loss/=len(train_loader)
        model.eval()
        val_loss=0
        with torch.no_grad():
            for images,masks,_ in tqdm(val_loader,desc='Val'):
                images,masks=images.to(device),masks.to(device)
                pred=model(images)
                l=loss(pred,masks)
                val_loss+=l.item()
        val_loss/=len(val_loader)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:4f}, Val Loss: {val_loss:.4f}')
        sche.step(val_loss)
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            torch.save(model.state_dict(),'best.pth')
    model.load_state_dict(torch.load('best.pth'))
    return model

def predict_and_submit(model,test_loader):
    model.eval()
    results=[]
    with torch.no_grad():
        for images,img_ids in tqdm(test_loader,desc='Predicting'):
            images=images.to(device)
            pred=model(images)
            pred=pred.cpu().numpy()
            for i,img_id in enumerate(img_ids):
                for class_id in range(1,num_classes+1):
                    mask=pred[i,class_id-1]>0.5
                    mask=mask.astype(np.uint8)
                    rle=rle_encoder(mask)
                    results.append({'ImageId':img_id,'EncodedPixels':rle,'ClassId':class_id})
    df_sub=pd.DataFrame(results)
    df_sub.to_csv('submission.csv',index=False)
    print(f'Submission saved to submission.csv')

def main():
    train_df=pd.read_csv('/kaggle/input/competitions/severstal-steel-defect-detection/train.csv')
    all_img_ids=train_df['ImageId'].unique()
    train_ids,val_ids=train_test_split(all_img_ids,test_size=0.2,random_state=7)
    train_df_split=train_df[train_df['ImageId'].isin(train_ids)]
    val_df_split=train_df[train_df['ImageId'].isin(val_ids)]
    train_dataset=SteelDataset(train_df_split,'/kaggle/input/competitions/severstal-steel-defect-detection/train_images',transform=train_transform)
    val_dataset=SteelDataset(val_df_split,'/kaggle/input/competitions/severstal-steel-defect-detection/train_images',transform=val_transform)
    train_loader=DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=4)
    val_loader=DataLoader(val_dataset,batch_size=8,shuffle=False,num_workers=4)
    test_dataset=TestDataset('/kaggle/input/competitions/severstal-steel-defect-detection/test_images')
    test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False,num_workers=4)
    net=UNet(n_classes=num_classes).to(device=device)
    net=train(net,train_loader,val_loader,epochs=10)
    predict_and_submit(net,test_loader)
    print('Done')

if __name__=='__main__':
    main()
    


        