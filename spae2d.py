import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class EncoderCNN2D(torch.nn.Module):
    def __init__(self,phase=True,latent_dim=3):
        self.e = 1.0e-10
        self.phase = phase
        super(EncoderCNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=2, padding=1)  # -> (16, 60, 60)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # -> (32, 30, 30)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # -> (64, 15, 15)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # -> (128, 8, 8)
        self.flatten = nn.Flatten()  # -> (128*8*8 = 8192)
        self.fc = nn.Linear(128 * 8 * 8, latent_dim)  # -> (3)


            
    def forward(self, x):
        x = F.relu(self.conv1(x))  # (2, 120, 120) -> (16, 60, 60)
        x = F.relu(self.conv2(x))  # -> (32, 30, 30)
        x = F.relu(self.conv3(x))  # -> (64, 15, 15)
        x = F.relu(self.conv4(x))  # -> (128, 8, 8)
        x = self.flatten(x)        # -> (8192)
        y = self.fc(x)      
        if self.phase:
            r = torch.norm(y[:,:2]+self.e,dim=1)
            r = r.tile((2,1)).T
            z1 = y[:,:2]/r
            z = torch.cat((z1,y[:,2:]),dim=1)
            return z
        else:
            return y  
        
class DecoderCNN2D(torch.nn.Module):
    def __init__(self,latent_dim=3):
        super(DecoderCNN2D, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)  # 入力: 3次元 -> (128, 8, 8)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # -> (64, 16, 16)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0)   # -> (32, 32, 32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0)   # -> (16, 64, 64)
        self.deconv4 = nn.ConvTranspose2d(16, 2, 4, stride=2, padding=2, output_padding=0)    # -> (2, 120, 120)

    def forward(self, x):
        x = self.fc(x)                 # -> (batch, 128*8*8)
        x = x.view(-1, 128, 8, 8)      # -> (batch, 128, 8, 8)
        x = F.relu(self.deconv1(x))   # -> (batch, 64, 16, 16)
        x = F.relu(self.deconv2(x))   # -> (batch, 32, 32, 32)
        x = F.relu(self.deconv3(x))   # -> (batch, 16, 64, 64)
        y = self.deconv4(x)  # -> (batch, 2, 120, 120)
        return y

class LatentSteper(torch.nn.Module):
    def __init__(self,latent_dim=3):
        super(LatentSteper, self).__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.01))
        self.lam = torch.nn.Parameter(torch.tensor([0.99]*(latent_dim-2)))

    def forward(self, input):
        z0 = input[:,[0]] * torch.cos(self.theta) - input[:,[1]] * torch.sin(self.theta)
        z1 = input[:,[0]] * torch.sin(self.theta) + input[:,[1]] * torch.cos(self.theta)
        z2 = input[:,2:] * self.lam
        z = torch.cat((z0,z1,z2),dim=1)

        return z
    
def to_polar(x):
    theta = torch.atan2(x[:, 1], x[:, 0])
    return theta


class DynamicalSystemDataset2D(Dataset):
    def __init__(self, data_dir,noise):
        """
        Parameters
        ----------
        data_dir:
        """
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.X0 = np.load('./limit_cycle_spiral2.npy')
        print(self.X0.shape)
        self.mu = np.mean(self.X0,axis=(0,2,3))
        self.std = np.std(self.X0,axis=(0,2,3))
        print(self.mu.shape)
        self.len = len(self.files)
        print(self.len)
        self.noise = noise

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file = os.path.join(self.data_dir,self.files[idx])
        data = np.load(file)
        for i in range(len(self.mu)):
            data[:,i] = data[:,i] - self.mu[i]
            data[:,i] = data[:,i]/self.std[i]
            data[:,i] += np.random.randn(data.shape[2],data.shape[3]) * self.noise
        return data

def argsWrite(p,log_file):
    f = open(log_file, 'a')
    f.write('-----------Prameter-----------\n')
    args = [(i, getattr(p, i)) for i in dir(p) if not '_' in i[0]]
    for i, j in args:
        f.write('{0}:{1}\n'.format(i, j))
    f.write('------------------------------\n\n')

def main():
    # Get arguments
    parser = argparse.ArgumentParser()
    # Parameter (experimental management)
    parser.add_argument('--ex_name', type=str, default='ex')  # experiment ID
    # Parameter (limitcycle)
    #parser.add_argument('--lc_name', type=str, default='SL')  # limitcycle(name),'SL','VP','HH','FHN3','FHNR'
    #parser.add_argument('--dt', type=float, default=0.001)  # Computation time step width for dynamical systems.
    #parser.add_argument('--noise_rate', type=float, default=0.5)  # Noise size
    #parser.add_argument('--num_rotation', type=int, default=3)  # Number of limit cycle turns
    #parser.add_argument('--data_interval', type=int, default=5)  # Parameters for how finely the data is taken. 
    # Parameter for thinning the data as the training data is huge when dt is small. 

    # モデル、学習に関するパラメータ
    parser.add_argument('--step_interval', type=int, default=10)
    parser.add_argument('--step_num', type=int, default=20)
    parser.add_argument('--latent_dim', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--epoch_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--w_step', type=float, default=0.1)
    parser.add_argument('--w_z1', type=float, default=2.0)
    parser.add_argument('--sw', type=float, default=1.0)
    parser.add_argument('--data_dir', type=str, default='./data')
    # モード
    #parser.add_argument('--check_data', action='store_true')
    #parser.add_argument('--train_traj_num', type=int, default=-1)
    parser.add_argument('--noise_level', type=float, default=0.0)

    args = parser.parse_args()

    ex_name = args.ex_name
    epoch_size = args.epoch_size
    step_num = args.step_num
    w_step = args.w_step
    w_z1 = args.w_z1
    sw = args.sw
    sw1 = 1.0
    sw2 = 1.0
    noise = args.noise_level

    # ログファイルの作成と結果ディレクトリの作成
    if not os.path.exists(f'./result'):
        os.mkdir(f'./result')
    if not os.path.exists(f'./result/{ex_name}'):
        os.mkdir(f'./result/{ex_name}')
    log_file = f'./result/{ex_name}/log.txt'
    f = open(log_file, 'w')
    f.write('Logging Start.\n\n')
    f.close()
    argsWrite(args, log_file)

    train_dataset = DynamicalSystemDataset2D(args.data_dir, noise=noise)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    device = 'cuda'
    #input_dim = lc_X0.shape[1]
    latent_dim = args.latent_dim
    enc = EncoderCNN2D(latent_dim=latent_dim)
    step = LatentSteper(latent_dim=latent_dim)
    dec = DecoderCNN2D(latent_dim=latent_dim)
    optimizer = torch.optim.Adam(list(enc.parameters())
                                    + list(dec.parameters())
                                    + list(step.parameters()),
                                    lr=args.lr)
    enc.to(device)
    step.to(device)
    dec.to(device)

    lc_X0 = np.load('./limit_cycle_spiral2.npy')
    mean = np.mean(lc_X0,axis=(0,2,3))
    std = np.std(lc_X0,axis=(0,2,3))
    
    for e in range(epoch_size):
        loss_vec = []
        enc.train()
        step.train()
        dec.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            xs = torch.Tensor(batch).to(device, dtype=torch.float)
            x = xs[:, 0, :]
            z = enc(x)
            y = dec(z)
            loss_recon = torch.nn.MSELoss()(x, y)
            for step_i in range(1, step_num):
                if step_i == 1:
                    step_z = step(z)
                else:
                    step_z = step(step_z)
                step_x = xs[:, step_i, :]
                enc_z = enc(step_x)
                if step_i == 1:
                    loss_step_phase = torch.nn.MSELoss()(step_z[:,:2], enc_z[:,:2])
                    loss_step_r = torch.nn.MSELoss()(step_z[:,2:], enc_z[:,2:])
                else:
                    mw = np.power(step_i, sw)
                    loss_step_phase += torch.nn.MSELoss()(step_z[:,:2], enc_z[:,:2])/mw
                    loss_step_r += torch.nn.MSELoss()(step_z[:,2:], enc_z[:,2:])/mw
            loss_z1 = torch.norm(torch.mean(z[:, :2], dim=0))
            loss_z2 = torch.norm(torch.mean(z[:, 2:]*z[:, 2:], dim=0)-1)

            #loss = loss_recon/d + loss_step * w_step + loss_z1 * w_z1
            loss = loss_recon + loss_step_phase * w_step * sw2 \
                + loss_step_r * w_step + loss_z1 * w_z1 * sw1 #+ loss_z2 * 0.1
            loss.backward()
            optimizer.step()
            loss_vec.append(np.array([loss.item(),
                                        loss_recon.item(),
                                        loss_step_phase.item(),
                                        loss_step_r.item(),
                                        loss_z1.item(),
                                        loss_z2.item()]))
        loss_vec = np.stack(loss_vec)
        print(e,
                np.mean(loss_vec[:, 0]),
                np.mean(loss_vec[:, 1]),
                np.mean(loss_vec[:, 2]),
                np.mean(loss_vec[:, 3]),
                np.mean(loss_vec[:, 4]),
                np.mean(loss_vec[:, 5]))
        
        sw = np.min([sw, np.mean(loss_vec[:, 2])])
        if np.mean(loss_vec[:, 2])<0.05 and np.mean(loss_vec[:, 4])<0.05: #default 0.01,0.05
            print('loss_z1を消去')
            sw1 = 0.00
            sw2 = 10.0
        #elif np.mean(loss_vec[:, 4])<0.1: #default 0.01,0.05f:
        #    print('loss_step_thetaを強化')
        #    sw1 = 1.0
        #    sw2 = 5.0
        else:
            sw1 = 1.0
            sw2 = 1.0
        
        if (e % 1) == 0:
            torch.save(enc.state_dict(),
                        f'./result/{ex_name}/enc.pth')
            torch.save(step.state_dict(),
                        f'./result/{ex_name}/step.pth')
            torch.save(dec.state_dict(),
                        f'./result/{ex_name}/dec.pth')
            
            torch.save(enc.state_dict(),
                        f'./result/{ex_name}/enc_e{str(e).zfill(3)}.pth')
            torch.save(step.state_dict(),
                        f'./result/{ex_name}/step_e{str(e).zfill(3)}.pth')
            torch.save(dec.state_dict(),
                        f'./result/{ex_name}/dec_e{str(e).zfill(3)}.pth')
        enc.eval()
        step.eval()
        dec.eval()

        def phase2state(p):
            z = [[np.cos(p),np.sin(p)]+[0.0]*(latent_dim-2)]
            z = torch.Tensor(z).to(device,dtype = torch.float)
            out = dec(z)
            x = out.detach().to('cpu').numpy()
            for j in range(2):
                x[:,j] = x[:,j]*std[j]
                x[:,j] = x[:,j] + mean[j]
            return x
        x1,y1 = 70,70
        x2,y2 = 100,100
        xs = []
        a = np.linspace(0,2*np.pi,lc_X0.shape[0])
        for _a in a:
            x = phase2state(_a)
            xs.append(x)
        xs = np.concatenate(xs)
        print(xs.shape)
        #plt.plot(lc_X0[:,0,x1],lc_X0[:,0,x2])
        #plt.plot(xs[:,0,x1],xs[:,0,x2])
        plt.plot(lc_X0[:,0,x1,y1],lc_X0[:,0,x2,y2])
        plt.plot(xs[:,0,x1,y1],xs[:,0,x2,y2])
        plt.savefig(f'./result/{ex_name}/X0_e{str(e).zfill(3)}.png')
        plt.close()

    

if __name__ == '__main__':
    main()