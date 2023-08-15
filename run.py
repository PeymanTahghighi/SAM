from data_utils import get_loader, cache_dataset, visualize_annotations
from model import SAM
import cv2
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch
import config
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def loss_ge(fg1, fg2, ge, landmark):
    a = np.array(list(range(0,config.BATCH_SIZE,1))).reshape(config.BATCH_SIZE, 1);
    sgi = torch.cosine_similarity(fg1[:,:,None,:], ge[:, None, :, :], dim = -1);
    #sgi_self = (F.normalize(fg2)@ge);

    sgi[a, list(range(0,37,1)),landmark] = 0;
    #sgi_self[a, list(range(0,37,1)),landmark] = 0;
    
    sgi_sort = torch.argsort(sgi,descending=True);
    #sgi_self_sort = torch.argsort(sgi_self,descending=True);

    hgi = sgi_sort[:,:,:config.NUM_NEGATIVES];
    #hgi_self = sgi_self_sort[:,:,:5];

    a = a.reshape(config.BATCH_SIZE, 1, 1);

    hgi = ge[a,hgi,:];
    #hgi_self = ge[a,:,hgi_self];
    
    nominator = torch.sum((fg1 *fg2)/config.TEMPERATURE, dim = 2);

    denominator = torch.sum((fg1[:,:,None,:] *hgi)/config.TEMPERATURE, dim = 3);
    #denominator_self = torch.sum(fg2.unsqueeze(dim=2) *(hgi_self/config.TEMPERATURE), dim = 3);
    loss = torch.sum(-nominator + torch.log(torch.exp(nominator) + torch.sum(torch.exp(denominator),dim=2) + 1e-6), dim = 1);
    return torch.mean(loss), sgi, #sgi_self;

def loss_le(fl1, fl2, le, sg, landmark):
    a = np.array(list(range(0,config.BATCH_SIZE,1))).reshape(config.BATCH_SIZE, 1);

    sl = torch.cosine_similarity(fl1[:,:,None,:], le[:, None, :, :], dim = -1);
    #sl = (F.normalize(fl1)@le);
    #sl_self = (F.normalize(fl2)@le);

    sg = sg.reshape(sg.shape[0], sg.shape[1], int(np.sqrt(sg.shape[2])), int(np.sqrt(sg.shape[2])))
    #sg_self = sg_self.reshape(sg_self.shape[0], sg_self.shape[1], int(np.sqrt(sg_self.shape[2])), int(np.sqrt(sg_self.shape[2])))

    sg = F.interpolate(sg, (int(np.sqrt(le.shape[1])), int(np.sqrt(le.shape[1]))), mode='nearest').reshape(sg.shape[0], sg.shape[1], sl.shape[2]);
    #sg_self = F.interpolate(sg_self, (int(np.sqrt(le.shape[2])), int(np.sqrt(le.shape[2])))).reshape(sg_self.shape[0], sg_self.shape[1], sl_self.shape[2]);

    sl = sl + sg;
    #sl_self = sl_self + sg_self;

    sl[a, list(range(0,37,1)),landmark] = 0;
    #sl_self[a, list(range(0,37,1)),landmark] = 0;
    
    sl = torch.argsort(sl,descending=True);
    #sl_self = torch.argsort(sl_self,descending=True);

    hl = sl[:,:,:config.NUM_NEGATIVES];
    #hl_self = sl_self[:,:,:5];

    a = a.reshape(config.BATCH_SIZE, 1, 1);

    hl = le[a,hl,:];
    #hl_self = le[a,:,hl_self];
    
    nominator = torch.sum((fl1 *fl2)/config.TEMPERATURE, dim = 2);
    

    denominator = torch.sum((fl1.unsqueeze(dim=2) *hl)/config.TEMPERATURE, dim = 3);
    denominator = torch.logsumexp(denominator, dim=-1);


    loss = torch.sum(-nominator + torch.log(torch.exp(nominator) + denominator), dim = 1);
    
    return torch.mean(loss);

def oneDtotwoD(oneD, size):
    r = torch.remainder(oneD,size).unsqueeze(dim=-1);
    d = torch.floor(torch.div(oneD,size)).unsqueeze(dim=-1);
    ret = torch.cat([d,r], dim=-1);
    return ret;

def MRE(fg, fl, ge, le, landmarks):
    sg = torch.cosine_similarity(fg[:,:,None,:], ge[:, None, :, :], dim = -1);
    sl = torch.cosine_similarity(fl[:,:,None,:], le[:, None, :, :], dim = -1);

    sg = sg.reshape(sg.shape[0], sg.shape[1], int(np.sqrt(sg.shape[2])), int(np.sqrt(sg.shape[2])))
    sg = F.interpolate(sg, (int(np.sqrt(le.shape[1])), int(np.sqrt(le.shape[1]))), mode='nearest').reshape(sg.shape[0], sg.shape[1], sl.shape[2]);

    size = sl.shape[2];
    sl = sl+sg;
    sl = torch.argmax(sl,dim=2);
    
    sl = oneDtotwoD(sl, np.sqrt(size));
    landmarks = oneDtotwoD(landmarks, np.sqrt(size));

    mre = torch.mean(torch.sum(torch.abs(sl - landmarks).float(), dim=-1));
    return mre;
    
def train(model, loader, optimizer, scalar):
    
    print(('\n' + '%10s'*2) %('Epoch', 'Loss'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_loss = [];

    for idx, img_annot in pbar:
        pair1, pair2, mask = img_annot;
        img1, landmark1, img2, landmark2, mask = pair1['image'].to('cuda'), pair1['landmarks'].to('cuda'), pair2['image'].to('cuda'), pair2['landmarks'].to('cuda'), mask.to('cuda');
        img1 = img1.permute(0, 3, 1, 2);
        img2 = img2.permute(0, 3, 1, 2);

        with torch.cuda.amp.autocast_mode.autocast():
            ge1, le1 = model(img1);
            ge2, le2 = model(img2);

            b = img1.shape[0];

            ge1 = ge1.reshape(b, ge1.shape[2]*ge1.shape[3], ge1.shape[1]);
            ge2 = ge2.reshape(b, ge2.shape[2]*ge2.shape[3], ge2.shape[1]);

            le1 = le1.reshape(b, le1.shape[2]*le1.shape[3], le1.shape[1]);
            le2 = le2.reshape(b, le2.shape[2]*le2.shape[3], le2.shape[1]);

            factor_g = ge1.shape[1]/(config.RESIZE**2);
            landmark1_g = (landmark1*factor_g).long();
            landmark2_g = (landmark2*factor_g).long();

            factor_l = le1.shape[1]/(config.RESIZE**2);
            landmark1_l = (landmark1*factor_l).long();
            landmark2_l = (landmark2*factor_l).long();

            a = np.array(list(range(0,b,1))).reshape(b, 1);
            fge1 = ge1[a,landmark1_g, :]; #(2,37,128)
            fge2 = ge2[a,landmark2_g, :];
            fle1 = le1[a,landmark1_l, :];
            fle2 = le2[a,landmark2_l, :];

            fge1 = (fge1 * mask.unsqueeze(dim = 2)).float();
            fge2 = (fge2 * mask.unsqueeze(dim = 2)).float();
            fle1 = (fle1 * mask.unsqueeze(dim = 2)).float();
            fle2 = (fle2 * mask.unsqueeze(dim = 2)).float();

            fle1 = F.normalize(fle1, p = 2, dim = 2)
            fle2 = F.normalize(fle2, p = 2, dim = 2)
            fge1 = F.normalize(fge1, p = 2, dim = 2)
            fge2 = F.normalize(fge2, p = 2, dim = 2)

            
            loss_ge1, sg12 = loss_ge(fge1, fge2, ge2, landmark2_g);
            loss_ge2, sg21 = loss_ge(fge2, fge1, ge1, landmark1_g);

            loss_le1 = loss_le(fle1, fle2, le2, sg12, landmark2_l);
            loss_le2 = loss_le(fle2, fle1, le1, sg21, landmark1_l);

            total_loss = loss_ge1 + loss_ge2 + loss_le1 + loss_le2;
        
        scalar.scale(total_loss).backward();
        epoch_loss.append(total_loss.item());
        scalar.step(optimizer);
        scalar.update();
        model.zero_grad(set_to_none = True);

        pbar.set_description(('%10s' + '%10.8g')%(epoch, np.mean(epoch_loss)));

    return np.mean(epoch_loss);

def valid(model, loader):
    
    print(('\n' + '%10s'*2) %('Epoch', 'MRE'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_mre = [];
    with torch.no_grad():
        for idx, img_annot in pbar:
            pair1, pair2, mask = img_annot;
            img1, landmark1, img2, landmark2, mask = pair1['image'].to('cuda'), pair1['landmarks'].to('cuda'), pair2['image'].to('cuda'), pair2['landmarks'].to('cuda'), mask.to('cuda');
            img1 = img1.permute(0, 3, 1, 2);
            img2 = img2.permute(0, 3, 1, 2);

            ge1, le1 = model(img1);
            ge2, le2 = model(img2);

            b = img1.shape[0];

            ge1 = ge1.reshape(b, ge1.shape[2]*ge1.shape[3], ge1.shape[1]);
            ge2 = ge2.reshape(b, ge2.shape[2]*ge2.shape[3], ge2.shape[1]);

            le1 = le1.reshape(b, le1.shape[2]*le1.shape[3], le1.shape[1]);
            le2 = le2.reshape(b, le2.shape[2]*le2.shape[3], le2.shape[1]);

            factor_g = ge1.shape[1]/(config.RESIZE**2);
            landmark1_g = (landmark1*factor_g).long();
            landmark2_g = (landmark2*factor_g).long();

            factor_l = le1.shape[1]/(config.RESIZE**2);
            landmark1_l = (landmark1*factor_l).long();
            landmark2_l = (landmark2*factor_l).long();

            a = np.array(list(range(0,b,1))).reshape(b, 1);
            fge1 = ge1[a,landmark1_g, :]; #(2,37,128)
            fge2 = ge2[a,landmark2_g, :];
            fle1 = le1[a,landmark1_l, :];
            fle2 = le2[a,landmark2_l, :];

            fge1 = (fge1 * mask.unsqueeze(dim = 2)).float();
            fge2 = (fge2 * mask.unsqueeze(dim = 2)).float();
            fle1 = (fle1 * mask.unsqueeze(dim = 2)).float();
            fle2 = (fle2 * mask.unsqueeze(dim = 2)).float();

            fle1 = F.normalize(fle1, p = 2, dim = 2)
            fle2 = F.normalize(fle2, p = 2, dim = 2)
            fge1 = F.normalize(fge1, p = 2, dim = 2)
            fge2 = F.normalize(fge2, p = 2, dim = 2)

            mre1 = MRE(fge1, fle1, ge2, le2, landmark2_l);
            mre2 = MRE(fge2, fle2, ge1, le1, landmark1_l);

            mre_total = mre1 + mre2;
            epoch_mre.append(mre_total.item());

            pbar.set_description(('%10s' + '%10.8g')%(epoch, np.mean(epoch_mre)));
    

    return np.mean(epoch_mre);

if __name__ == "__main__":

    def unstable_softmax(logits):
        exp = torch.exp(logits - torch.max(logits))
        return exp / torch.sum(exp)

    print(unstable_softmax(torch.tensor([1000., 0.])).numpy())  # prints [ nan, 0.]
    # a = torch.from_numpy(np.arange(0,2*5*100,1).reshape(2,5,100)).float();
    # b = torch.from_numpy(np.arange(0,2*3*10,1).reshape(2,3,10)).float();
    # b[[[0],[1]], [0,1,2], [[1,2,3], [0,5,6]]] = 0;
    # b = torch.argsort(b, dim = -1, descending=False )
    # print(b);
    #feats[:,None,:], feats[None,:,:], dim=-1
    #e = torch.cosine_similarity(c[:,:,None, :],b[:,None,:,:],dim=-1);
    #f = torch.cosine_similarity(c[:,None,:,:],b[:,:,None, :],dim=-1);

    #d = e==f;
    #print(e);
    #print(f);
    #only do it once
    #cache_dataset();
    a = torch.randn(2,2,625);
    a[[[0]],[0,1],[[0],[1]]] = 0;
    k = np.broadcast_arrays([[0]],[0,1],[0,1]);


    RESUME = False;
    train_loader, test_loader = get_loader();
    model = SAM();
    if RESUME is True:
        ckpt = torch.load('checkpoint.ckpt');
        model.load_state_dict(ckpt['model']);
    
    model.to('cuda');
    scalar = torch.cuda.amp.grad_scaler.GradScaler();
    optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE);
    summary_writer = SummaryWriter(f'exp/{config.EXP_NAME}')
    best_mre = 1e9;
    checkpoint_interval = 5;
    start_epoch = 0;

    if RESUME is True:
        optimizer.load_state_dict(ckpt['optimizer']);
        best_mre = ckpt['mre'];
        scalar.load_state_dict(ckpt['scalar']);
        start_epoch = ckpt['epoch'];
        print(f'Resuming from {start_epoch}');
        
    for epoch in range(start_epoch, 1000):
        model.train();
        loss = train(model, train_loader, optimizer, scalar);
        model.eval();
        mre = valid(model, test_loader);
        if mre < best_mre:
            print(f'New best model found: {mre}');
            ckpt = {'model': model.state_dict()};
            torch.save(ckpt, 'best_model.ckpt');
            best_mre = mre;
        elif epoch % checkpoint_interval == 0:
            ckpt = {
                'model': model.state_dict(),
                'epoch': epoch+1,
                'optimizer': optimizer.state_dict(),
                'scalar': scalar.state_dict(),
                'mre': best_mre
            }
            torch.save(ckpt, 'checkpoint.ckpt');
        summary_writer.add_scalar('loss', loss, epoch);
        summary_writer.add_scalar('mre', mre, epoch);
    

        


