import numpy as np
from test_data import test_dataset
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm

dataset_path = '' ##gt_path

dataset_path_pre = ''  ##pre_salmap_path

test_datasets = ['']     ##test_datasets_name

for dataset in test_datasets:
    sal_root = dataset_path_pre  +'/'
    gt_root = dataset_path  +'/GT/'
    test_loader = test_dataset(sal_root, gt_root)
    mae,fm,sm,em,wfm= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm()
    for i in range(test_loader.size):
        print ('predicting for %d / %d' % ( i + 1, test_loader.size))
        sal, gt = test_loader.load_data()
        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)
        if res.max() == res.min():
            res = res/255
        else:
            res = (res - res.min()) / (res.max() - res.min())
        mae.update(res, gt)
        sm.update(res,gt)
        fm.update(res, gt)
        em.update(res,gt)
        wfm.update(res,gt)

    MAE = mae.show()
    maxf,meanf,_,_ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(dataset, MAE, maxf,meanf,wfm,sm,em))