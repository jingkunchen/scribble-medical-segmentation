from networks.efficientunet import Effi_UNet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS, UNet_CCT, UNet_CCT_3H, UNet_Trans, UNet_Matching,UNet_Matching_All_Queue


def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_matching":
        net = UNet_Matching(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_matching_all_queue":
        net = UNet_Matching_All_Queue(in_chns=in_chns, class_num=class_num).cuda()
        
    elif net_type == "unet_cct_3h":
        net = UNet_CCT_3H(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "UNet_Trans":
        net = UNet_Trans(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    else:
        net = None
    return net