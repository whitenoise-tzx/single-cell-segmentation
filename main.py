
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim, multiprocessing

import dataset
from UNet import Unet,resnet34_unet
from attention_unet import AttU_Net
from channel_unet import myChannelUnet
from r2unet import R2U_Net
from segnet import SegNet
from unetpp import NestedUNet
from fcn import get_fcn8s
from dataset import *
from metrics import *
from metrics_3 import *
from torchvision.transforms import transforms
from plot import loss_plot
from plot import metrics_plot
from torchvision.models import vgg16
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=41)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='r2unet',
                       help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument('--dataset', default='T47D',  # dsb2018_256
                       help='dataset name:isbiCell/NMuMg/T47D/HK2_DIC')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    parse.add_argument("--num_classes", type=int, default=3)
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    if args.arch == 'UNet':
        model = Unet(1, 3).to(device)
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(pretrained=False).to(device)
    if args.arch == 'unet++':
        args.deepsupervision = True
        model = NestedUNet(args,1,3).to(device)
    if args.arch =='Attention_UNet':
        model = AttU_Net(1,3).to(device)
    if args.arch == 'segnet':
        model = SegNet(1,3).to(device)
    if args.arch == 'r2unet':
        model = R2U_Net(1,3).to(device)
    #if args.arch == 'fcn32s':
         #model = get_fcn32s(3).to(device)
    if args.arch == 'myChannelUnet':
        model = myChannelUnet(1,3).to(device)
    if args.arch == 'fcn8s':
        assert args.dataset !='esophagus' ,"fcn8s模型不能用于数据集esophagus，因为esophagus数据集为80x80，经过5次的2倍降采样后剩下2.5x2.5，分辨率不能为小数，建议把数据集resize成更高的分辨率再用于fcn"
        model = get_fcn8s(3).to(device)
    if args.arch == 'cenet':
        from cenet import CE_Net_
        model = CE_Net_().to(device)
    return model

def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    if args.dataset == 'isbiCell':
        train_dataset = IsbiCellDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = IsbiCellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = IsbiCellDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'NMuMg':
        train_dataset = NMuMgDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = NMuMgDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = NMuMgDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'T47D':
        train_dataset = T47DDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = T47DDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = T47DDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'HK2_DIC':
        train_dataset = HK2_DICDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = HK2_DICDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = HK2_DICDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders,val_dataloaders,test_dataloaders

def val(model,best_iou,val_dataloaders):
    model= model.eval()
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        pixel_accuracy_total= 0
        #CCA_total=0
        num = len(val_dataloaders)  #验证集图片的总数
        print('验证集图片总数:',num)
        for x, _,pic,mask in val_dataloaders:
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            img_gt = Image.open(mask[0])
            img_gt = np.asarray(img_gt)
            pred=np.argmax(img_y,axis=0)
            pixel_accuracy_total+= pixel_accuracy(pred,img_gt)
            miou_total += mean_IU(pred,img_gt)
            #CCA_total += accuracy_score(pred,img_gt)
            if i < num:i+=1   #处理验证集下一张图
        torch.save(model.state_dict(), r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(
            args.dataset) + '_' + str(args.epoch) + '.pth')
        aver_pixel_accuracy=pixel_accuracy_total/num
        aver_iou = miou_total / num
        #aver_CCA = CCA_total/num
        print('aver_pixel_accuracy=%f, Miou=%f' % (aver_pixel_accuracy,aver_iou))
        logging.info('aver_pixel_accuracy=%f, Miou=%f' % (aver_pixel_accuracy,aver_iou))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
        return best_iou,aver_iou,aver_pixel_accuracy


def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    best_iou,aver_iou,aver_CCA,aver_pixel_accuracy= 0,0,0,0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    pixel_accuracy_list = []
    #CCA_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y,_,mask in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    output = output.to(device)
                    loss += criterion(output, torch.argmax(labels, dim=1))
                loss /= len(outputs)
            else:
                output = model(inputs)
                output = output.to(device)
                #print(output.shape)
                loss = criterion(output, torch.argmax(labels, dim=1))
            if threshold!=None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() #item()取标量tensor

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)
        best_iou,aver_iou,aver_pixel_accuracy = val(model,best_iou,val_dataloader)
        iou_list.append(aver_iou)
        pixel_accuracy_list.append(aver_pixel_accuracy)
        #CCA_list.append(aver_CCA)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou',iou_list)
    metrics_plot(args,'pixel accuracy',pixel_accuracy_list)
    #metrics_plot(args, 'CCA', CCA_list)
    return model

def test(val_dataloaders,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        dir = os.path.join(r'./saved_predict',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    model.load_state_dict(torch.load(r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()
    total=0
    correct=0
    #plt.ion() #开启动态模式
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        pixel_accuracy_total = 0
        #CCA_total = 0
        num = len(val_dataloaders)  #验证集图片的总数
        for pic,_,pic_path,mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            #img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            img_gt = Image.open(mask_path[0])
            img_gt = np.asarray(img_gt)
            pred = np.argmax(predict, axis=0)
            pixel_acc=pixel_accuracy(pred, img_gt)
            pixel_accuracy_total += pixel_acc
            iou=mean_IU(pred, img_gt)
            miou_total += iou
            #CCA_=accuracy_score(pred, img_gt)
            #CCA_total += CCA_
            #_, predicted = torch.max(torch.tensor(predict), dim=1)
            #total += mask_path[0].size(0)
            #correct += (predicted == mask_path[0]).sum().item()
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]), cmap='Greys_r')
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(pred,cmap = plt.get_cmap('gray'))
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(img_gt, cmap='Greys_r')
            if save_predict == True:
                if args.dataset == 'driveEye':
                    saved_predict = dir + '/' + mask_path[0].split('\\')[-1]
                    saved_predict = '.'+saved_predict.split('.')[1] + '.tif'
                    plt.savefig(saved_predict)
                else:
                    #plt.savefig(dir + '/' + mask_path[0].split('\\')[-1])
                    plt.savefig(dir +'/'+ mask_path[0].split('/')[-1])#分割路径并取最后一项
            #plt.pause(0.01)
            print('iou={},pixel_accuracy={}'.format(iou,pixel_acc))
            if i < num:i+=1   #处理验证集下一张图
        #plt.show()
        #print('accuracy on test set: %d %% ' % (100 * correct / total))
        print('Miou=%f,aver_pixel_accuracy=%f' % (miou_total/num,pixel_accuracy_total/num))
        logging.info('Miou=%f,aver_hd=%f' % (miou_total/num,pixel_accuracy_total/num))

if __name__ =="__main__":
    palette = [[0], [1], [2]]
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
        transforms.Normalize((0.5,), (0.5,))
    ])
    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('**************************')
    model = getModel(args)
    #model.load_state_dict(torch.load('./saved_model/UNet_1_T47D_21.pth'))  # 再加载网络的参数
    #model = model.to(device)
    #print("load success")
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)