import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt,cv2 
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(128)

device='cuda'

transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

class Brain_stroke(Dataset):
    def __init__(self):
        super(Brain_stroke,self).__init__()
        
        self.images_dir_path="C:/Users/rg528/Downloads/brain_stroke_detection_using_deep_cnn/brain_stroke_detection/train/images"
        self.masks_dir_path='C:/Users/rg528/Downloads/brain_stroke_detection_using_deep_cnn/brain_stroke_detection/train/labels'
        self.images=os.listdir(self.images_dir_path)
        self.masks=os.listdir(self.masks_dir_path)
       

    def __len__(self):

        return len(self.masks)

        
    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.images_dir_path, self.images[idx])).convert("RGB")
        img_np = np.array(img)

        img_ten = transform(img)

        h, w = img_np.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        with open(os.path.join(self.masks_dir_path, self.images[idx]).replace(".jpg", ".txt")) as f:

            for line in f.readlines():

                values = list(map(float, line.split()))
                class_id = int(values[0]) + 1 

                polygon = np.array(values[1:], dtype=np.float32).reshape(-1, 2)

                polygon[:, 0] *= w
                polygon[:, 1] *= h

                polygon = polygon.astype(np.int32)

                cv2.fillPoly(mask, [polygon], class_id)

    
        mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_NEAREST)

        mask = torch.from_numpy(mask)

        return img_ten.float(), mask.long()
        

ob=Brain_stroke()
#print(ob[10][1].unique())

#plt.imshow(ob[23][1])
#plt.savefig('./Desktop/test.jpg')

train_len=int(0.8*len(ob))
val_len=len(ob)-train_len
train_data,test_data=random_split(ob,[train_len,val_len])
dl_train=DataLoader(train_data,batch_size=4,shuffle=True,drop_last=True)
dl_val=DataLoader(test_data,batch_size=4,shuffle=True,drop_last=True)

#model

def conv(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))

def up_conv(in_channels,out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2),
                         nn.ReLU(inplace=True))
    
from torchvision.models import vgg16_bn

class Unet(nn.Module):
    def __init__(self,pretrained=True,out_channels=3):
        super(Unet,self).__init__()

        self.encoder=vgg16_bn(pretrained=pretrained).features
        self.block1=nn.Sequential(*self.encoder[:6])
        self.block2=nn.Sequential(*self.encoder[6:13])
        self.block3=nn.Sequential(*self.encoder[13:20])
        self.block4=nn.Sequential(*self.encoder[20:27])
        self.block5=nn.Sequential(*self.encoder[27:34])

        self.bottleneck=nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck=conv(512,1024)
        
        self.up_conv6=up_conv(1024,512)
        self.conv6=conv(512+512,512)
        self.up_conv7=up_conv(512,256)
        self.conv7=conv(256+512,256)
        self.up_conv8=up_conv(256,128)
        self.conv8=conv(128+256,128)
        self.up_conv9=up_conv(128,64)
        self.conv9=conv(64+128,64)
        self.up_conv10=up_conv(64,32)
        self.conv10=conv(32+64,32)
        self.conv11=nn.Conv2d(32,out_channels,kernel_size=1)


    def forward(self,x):
        block1=self.block1(x)
        block2=self.block2(block1)
        block3=self.block3(block2)
        block4=self.block4(block3)
        block5=self.block5(block4)
        
        bottleneck=self.bottleneck(block5)
        x=self.conv_bottleneck(bottleneck)

        x=self.up_conv6(x)
        x=torch.cat([x,block5],dim=1)
        x=self.conv6(x)

        x=self.up_conv7(x)
        x=torch.cat([x,block4],dim=1)
        x=self.conv7(x)

        x=self.up_conv8(x)
        x=torch.cat([x,block3],dim=1)
        x=self.conv8(x)

        x=self.up_conv9(x)
        x=torch.cat([x,block2],dim=1)
        x=self.conv9(x)

        x=self.up_conv10(x)
        x=torch.cat([x,block1],dim=1)
        x=self.conv10(x)

        x=self.conv11(x)

        return x

model=Unet().to(device)
#print(model)
#weight_tensor=torch.tensor([0.1,0.9,0.5]).to(device)
c_l=nn.CrossEntropyLoss()
opt=optim.AdamW(model.parameters())

def dice(pred,targ):
    prob1=torch.softmax(pred,dim=1)
    lab1=torch.argmax(prob1,dim=1)
    intersection = (2*(lab1.sum()*targ.sum()))/(lab1.sum()+targ.sum())
    return intersection

def train(x,y,model,opt,loss):
    model.train()
    opt.zero_grad()
    x=x.to(device)
    y=y.to(device)
    pred=model(x)
    dice_loss=1-dice(pred,y)
    loss=c_l(pred,y)
    total_loss=loss+dice_loss
    total_loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def acc(x,y,model):
    model.eval()
    x=x.to(device)
    y=y.to(device)
    pred=model(x)
    prob=torch.softmax(pred,dim=1)
    lab=torch.argmax(prob,dim=1)
    correct_label=(lab==y).float().mean()
    return correct_label.item()

train_loss=[];val_acc=[]
for epoch in range(10):
    print(f'epoch: {epoch+1}')
    train_loss_epoch=[];val_acc_epoch=[]
    
    for img,mask in dl_train:
        l=train(img,mask,model,opt,c_l)
        train_loss_epoch.append(l)
    
    for img,mask in dl_val:
        accuracy1=acc(img,mask,model)
        val_acc_epoch.append(accuracy1)
    
    train_loss.append(np.mean(train_loss_epoch))
    val_acc.append(np.mean(val_acc_epoch))
    print(train_loss[-1])

    torch.cuda.empty_cache()    
print(f'train loss: {np.mean(train_loss)} val_acc: {np.mean(val_acc)}')

torch.save(model.state_dict(),'C:/Users/rg528/Downloads/brain_stroke_detection_using_deep_cnn/brain_stroke_detection/brain_stroke_model.pth')

