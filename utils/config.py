# coding: utf-8
import os



duts_root_test ='/home/asus/Datasets/binary_segmentation/rgb_sod/DUTS/DUTS-TE'
ecssd_root_test ='/home/asus/Datasets/binary_segmentation/rgb_sod/ECSSD'
hku_is_root_test ='/home/asus/Datasets/binary_segmentation/rgb_sod/HKU-IS'
dut_omron_root_test ='/home/asus/Datasets/binary_segmentation/rgb_sod/DUT_OMRON'
pascal_s_root_test ='/home/asus/Datasets/binary_segmentation/rgb_sod/PASCAL-S'

CAMO_root_test ='/home/asus/Datasets/binary_segmentation/cod/TestDataset/TestDataset/CAMO'
NC4K_root_test ='/home/asus/Datasets/binary_segmentation/cod/TestDataset/TestDataset/NC4K'
CHAMELEON_root_test ='/home/asus/Datasets/binary_segmentation/cod/TestDataset/TestDataset/CHAMELEON'
COD10K_root_test ='/home/asus/Datasets/binary_segmentation/cod/TestDataset/TestDataset/COD10K'

CVC_300_root_test ='/home/asus/Datasets/binary_segmentation/2D_medical/polyp/TestDataset/TestDataset/CVC-300'
CVC_ClinicDB_root_test ='/home/asus/Datasets/binary_segmentation/2D_medical/polyp/TestDataset/TestDataset/CVC-ClinicDB'
CVC_ColonDB_root_test ='/home/asus/Datasets/binary_segmentation/2D_medical/polyp/TestDataset/TestDataset/CVC-ColonDB'
ETIS_LaribPolypDB_root_test ='/home/asus/Datasets/binary_segmentation/2D_medical/polyp/TestDataset/TestDataset/ETIS-LaribPolypDB'
Kvasir_root_test ='/home/asus/Datasets/binary_segmentation/2D_medical/polyp/TestDataset/TestDataset/Kvasir'

SBU_root_Test ='/home/asus/Datasets/binary_segmentation/shadow_detection/SBU/SBU-shadow/SBU-Test_rename'
ucf_root_test = '/home/asus/Datasets/binary_segmentation/shadow_detection/UCF/UCF'
ISTD_root_test = '/home/asus/Datasets/binary_segmentation/shadow_detection/ISTD/test'


trans10k_root_test_easy = '/home/asus/Datasets/binary_segmentation/transparent/trans10k/test(1)/test/easy'
trans10k_root_test_hard = '/home/asus/Datasets/binary_segmentation/transparent/trans10k/test(1)/test/hard'
trans10k_root_test_all = '/home/asus/Datasets/binary_segmentation/transparent/trans10k/test(1)/test/all'

dutrgbd_root_test = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/DUT-RGBD/test_data'
DUTLF_V2_root_test = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/DUTLF-V2'
njud_root_test = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/NJUD_test'
nlpr_root_test = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/NLPR_test'
stere_root_test = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/STERE'
sip_root_test = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/SIP'
rgbd135_root_test = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/RGBD135'
ssd_root_test = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/SSD'
lfsd_root_test = '/home/asus/Datasets/binary_segmentation/RGBD_SOD_Datasets/LFSD'

EORSSD_root_test = '/home/asus/Datasets/binary_segmentation/ORSI-SOD/EORSSD/testset'
ORSSD_root_test = '/home/asus/Datasets/binary_segmentation/ORSI-SOD/ORSSD/testset'
ORSI_4199_root_test = '/home/asus/Datasets/binary_segmentation/ORSI-SOD/ORSI-4199/testset'

CUHK_root_test = '/home/asus/Datasets/binary_segmentation/DBD/test_data/CUHK'
DUT_root_test = '/home/asus/Datasets/binary_segmentation/DBD/test_data/DUT'

GDD_root_test = '/home/asus/Datasets/binary_segmentation/Glass/GDD/test'

MSD_root_test = '/home/asus/Datasets/binary_segmentation/Mirror/MSD/test'





duts = os.path.join(duts_root_test)
ecssd = os.path.join(ecssd_root_test)
hku_is = os.path.join(hku_is_root_test)
dut_omron = os.path.join(dut_omron_root_test)
pascal_s = os.path.join(pascal_s_root_test)

CAMO = os.path.join(CAMO_root_test)
NC4K = os.path.join(NC4K_root_test)
CHAMELEON = os.path.join(CHAMELEON_root_test)
COD10K = os.path.join(COD10K_root_test)

CVC_300 = os.path.join(CVC_300_root_test)
CVC_ClinicDB = os.path.join(CVC_ClinicDB_root_test)
CVC_ColonDB = os.path.join(CVC_ColonDB_root_test)
ETIS_LaribPolypDB = os.path.join(ETIS_LaribPolypDB_root_test)
Kvasir = os.path.join(Kvasir_root_test)

SBU = os.path.join(SBU_root_Test)
ucf = os.path.join(ucf_root_test)
ISTD = os.path.join(ISTD_root_test)

trans10k_easy = os.path.join(trans10k_root_test_easy)
trans10k_hard = os.path.join(trans10k_root_test_hard)
trans10k_all = os.path.join(trans10k_root_test_all)

dutrgbd = os.path.join(dutrgbd_root_test)
DUTLF_V2 = os.path.join(DUTLF_V2_root_test)
njud = os.path.join(njud_root_test)
nlpr = os.path.join(nlpr_root_test)
stere = os.path.join(stere_root_test)
sip = os.path.join(sip_root_test)
rgbd135 = os.path.join(rgbd135_root_test)
ssd = os.path.join(ssd_root_test)
lfsd = os.path.join(lfsd_root_test)

EORSSD = os.path.join(EORSSD_root_test)
ORSSD = os.path.join(ORSSD_root_test)
ORSI_4199 = os.path.join(ORSI_4199_root_test)

CUHK = os.path.join(CUHK_root_test)
DUT = os.path.join(DUT_root_test)

GDD = os.path.join(GDD_root_test)

MSD = os.path.join(MSD_root_test)


