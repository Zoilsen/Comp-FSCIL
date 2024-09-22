# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F


def base_train(model, trainloader, optimizer, scheduler, epoch, args, is_base=False):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)


    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]


        logits = model(data, is_base=is_base, epoch=epoch)
        logits = logits[:, :args.base_class]
        
        if args.map_metric_option != 'none':
            map_logits = model.map_metric_forward(is_base=True)
            map_logits = map_logits[:, :args.base_class]
       
        train_label = train_label.to(torch.int64)
        
        loss = F.cross_entropy(logits, train_label)
        
        acc = count_acc(logits, train_label)

        if args.map_metric_option != 'none':
            map_metric_loss = F.cross_entropy(map_logits, train_label)
            loss = args.backbone_feat_cls_weight * loss + args.map_metric_cls_w * map_metric_loss


        if args.primitive_recon_cls_weight != 0.0 and epoch >= args.begin_epoch:
            bkb_map = model.end_points['final_map'] # [b, c, h, w]

            b, c, h, w = bkb_map.shape
            bkb_map = bkb_map.view(b, c, h*w)
            
            assert args.map_metric_option == 'cka'
            map_proto = model.fc_map_base[:args.base_class]#.detach() # [base_class, c, hp, wp]
            _, _, hp, wp = map_proto.shape
            map_proto = map_proto.view(args.base_class, c, hp*wp)
            
            bkb_map_mean = bkb_map.mean(dim=1) # [b, h*w]
            map_proto_mean = map_proto.mean(dim=1) # [base_class, hp*wp]

            bkb_map = bkb_map - bkb_map_mean.unsqueeze(1) # [b, c, h*w]
            map_proto = map_proto - map_proto_mean.unsqueeze(1) # [base_class, c, hp*wp]

            map_proto = map_proto.permute(0, 2, 1) # [base_class, hp*wp, c]
            map_proto = map_proto.reshape(args.base_class * hp * wp, c) # [base_class * hp * wp, c]

            bkb_map = bkb_map.permute(0, 2, 1) # [b, h*w, c]
            bkb_map = bkb_map.reshape(b*h*w, c) # [b * h * w, c]

            sims = - torch.cdist(map_proto, map_proto) ** 2 # [base_class * hp * wp, base_class * hp * wp]

            sims = sims.view(args.base_class, hp*wp, args.base_class, hp*wp) # [base_class, hp*wp, base_class, hp * wp]
            
            map_label = torch.tensor(range(args.base_class)).cuda() # [base_class]
            sim_mask = F.one_hot(map_label, args.base_class) # [base_class, base_class]
            sim_mask = sim_mask.unsqueeze(1).unsqueeze(-1) # [base_class, 1, base_class, 1]

            other_sims = sims - sim_mask * 9999 # [base_class, hp*wp, base_class, hp * wp]
            other_sims = other_sims.reshape(args.base_class * hp * wp, args.base_class * hp * wp)
            
            ## soft recon ##
            atten = torch.softmax(other_sims * args.aux_param, dim=-1).detach() # [base_class * hp * wp, base_class * hp * wp]

            recon = torch.matmul(atten, map_proto) # [base_class * hp * wp, c]

            recon = recon.reshape(args.base_class, hp * wp, c).permute(0, 2, 1).unsqueeze(0) # [1, base_class, c, hp*wp]
            bkb_map = bkb_map.reshape(b, h*w, c).permute(0, 2, 1).unsqueeze(1) # [b, 1, c, h*w]

            cross_norm = torch.norm(torch.matmul(bkb_map.permute(0, 1, 3, 2), recon), dim=[2,3]) ** 2 # [b, base_class]
            bkb_map_norm = torch.norm(torch.matmul(bkb_map.permute(0, 1, 3, 2), bkb_map), dim=[2,3]) # [b, 1]
            recon_norm = torch.norm(torch.matmul(recon.permute(0, 1, 3, 2), recon), dim=[2,3]) # [1, base_class]

            prim_recon_cls_logits = cross_norm / (bkb_map_norm * recon_norm + 0.000001) # [b, base_class]
            prim_recon_cls_logits = prim_recon_cls_logits * model.fc_map_temperature#.detach()
            
            prim_recon_cls_loss = F.cross_entropy(prim_recon_cls_logits, train_label)
            loss = args.primitive_recon_cls_weight * prim_recon_cls_loss + loss


        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    if args.not_data_init:
        model = model.eval()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                  num_workers=args.num_workers, pin_memory=True, shuffle=False)

        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                pass
        
        return model

    else:
        # replace fc.weight with the embedding average of train data
        model = model.eval()

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                  num_workers=args.num_workers, pin_memory=True, shuffle=False)
        original_transform = trainloader.dataset.transform
        trainloader.dataset.transform = transform
        embedding_list = []
        if args.map_metric_option != 'none':
            map_list = []

        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                model.mode = 'encoder'
                embedding = model(data)

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

                if args.map_metric_option != 'none':
                    feat_map = model.end_points['final_map'] # [b, c, h, w]
                    map_list.append(feat_map.cpu())


        embedding_list = torch.cat(embedding_list, dim=0)
        if args.map_metric_option != 'none':
            map_list = torch.cat(map_list, dim=0)

        label_list = torch.cat(label_list, dim=0)

        proto_list = []
        if args.map_metric_option != 'none':
            map_proto_list = []

        for class_index in range(args.base_class):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)
            
            if args.map_metric_option != 'none':
                map_this = map_list[data_index.squeeze(-1)]
                map_this = map_this.mean(0)
                map_proto_list.append(map_this)

        proto_list = torch.stack(proto_list, dim=0)
        model.fc.weight.data[:args.base_class] = proto_list

        if args.map_metric_option != 'none':
            model.fc_map.data[:args.base_class] = torch.stack(map_proto_list, dim=0)
           
        trainloader.dataset.transform = original_transform

        return model



def test(model, testloader, epoch, args, session, is_base=False):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager(); va = Averager()
    if args.in_domain_feat_cls_weight != 0.0:
        ind_va = Averager(); cmb_va = Averager()

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data, is_base=is_base)

            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label.to(torch.int64))
            acc = count_acc(logits, test_label)
            

            if args.in_domain_feat_cls_weight != 0.0:
                if args.map_metric_option != 'none':
                    map_logits = model.map_metric_forward(is_base=is_base)
                    in_domain_logits = map_logits

                in_domain_logits = in_domain_logits[:, :test_class]

                in_domain_acc = count_acc(in_domain_logits, test_label)
                combine_acc = count_acc(logits + in_domain_logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            if args.in_domain_feat_cls_weight != 0.0:
                ind_va.add(in_domain_acc); cmb_va.add(combine_acc)

        vl = vl.item()
        va = va.item()
        if args.in_domain_feat_cls_weight != 0.0:
            ind_va = ind_va.item(); cmb_va = cmb_va.item()

    if args.in_domain_feat_cls_weight == 0.0:
        #print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))#; exit()
        return vl, va
    else:
        #print('epo {}, test, loss={:.4f} acc={:.4f} (ind {:.4f}, cmb {:.4f})'.format(epoch, vl, va, ind_va, cmb_va))#; exit()
        return vl, va, ind_va, cmb_va


def test_all_sessions(model, testloader, epoch, args, total_session):
    # given the input of the last session, output acc of all sessions
    model = model.eval()
    
    vls = []; vas = []; num_samples = []
    if args.in_domain_feat_cls_weight != 0.0:
        ind_vas = []; cmb_vas = []

    for i in range(total_session + 1): # the last session is for all the novel classes
        vls.append([]); vas.append([]); num_samples.append([])
        if args.in_domain_feat_cls_weight != 0.0:
            ind_vas.append([]); cmb_vas.append([])

    def SelectFromDefault(data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = torch.where(i == targets)[0]
            if data_tmp == []:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = torch.vstack((data_tmp, data[ind_cl]))
                targets_tmp = torch.hstack((targets_tmp, targets[ind_cl]))
        return data_tmp, targets_tmp


    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data, is_base=True if args.not_data_init else False)

            if args.in_domain_feat_cls_weight != 0.0:
               if args.map_metric_option != 'none':
                   map_logits = model.map_metric_forward(is_base=True if args.not_data_init else False, aaa=False)
                   in_domain_logits = map_logits

            for k in range(total_session + 1): # the last session is for all the novel classes
                test_class = args.base_class + k * args.way
                if k < total_session:
                    test_logits = logits[:, :test_class]
                else:
                    test_logits = logits[:, args.base_class:]
                
                if k < total_session:
                    session_classes = np.arange(test_class)
                else:
                    session_classes = np.arange(args.base_class, args.base_class + (total_session-1) * args.way)

                session_logits, session_label = SelectFromDefault(test_logits, test_label, session_classes)
                
                if k == total_session:
                    session_label = session_label - args.base_class

                if len(session_label) > 0:
                    session_loss = F.cross_entropy(session_logits, session_label.to(torch.int64))
                    session_acc = count_acc(session_logits, session_label)
                    vls[k].append(session_loss); vas[k].append(session_acc); num_samples[k].append(len(session_label))

                if args.in_domain_feat_cls_weight != 0.0:
                    if k < total_session:
                        in_domain_test_logits = in_domain_logits[:, :test_class]
                    else:
                        in_domain_test_logits = in_domain_logits[:, args.base_class:]
                                       
                    in_domain_session_logits, session_label = SelectFromDefault(in_domain_test_logits, test_label, session_classes)

                    if k == total_session:
                        session_label = session_label - args.base_class

                    if len(session_label) > 0:
                        in_domain_session_acc = count_acc(in_domain_session_logits, session_label)
                        combine_session_acc = count_acc(in_domain_session_logits + session_logits, session_label)
                        ind_vas[k].append(in_domain_session_acc); cmb_vas[k].append(combine_session_acc)
        
        def get_average_results(session_vs, session_samples):
            assert len(session_vs) == len(session_samples)
            total_vs = 0.0; total_samples = 0
            for i in range(len(session_samples)):
                total_vs += session_vs[i] * session_samples[i]
                total_samples += session_samples[i]
            return total_vs / total_samples

        for i in range(total_session + 1): # the last session is for all the novel classes
            vls[i] = float(get_average_results(vls[i], num_samples[i]))
            vas[i] = float(get_average_results(vas[i], num_samples[i]))
            if args.in_domain_feat_cls_weight != 0.0:
                ind_vas[i] = float(get_average_results(ind_vas[i], num_samples[i]))
                cmb_vas[i] = float(get_average_results(cmb_vas[i], num_samples[i]))


    if args.in_domain_feat_cls_weight == 0.0:
        return vls, vas
    else:
        return vls, vas, ind_vas, cmb_vas

