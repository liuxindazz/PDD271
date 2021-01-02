from torch import nn
import torch
import torch.nn.functional as F
from core import densenet
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Spatial_attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self,feature_map,K = 512):
        """
        :param feature_map: feature map in level L
        :param decoder_dim: size of decoder's RNN
        """
        super(Spatial_attention, self).__init__()
        _,self.C,self.H,self.W = feature_map.shape
        # print("C:{},H:{},W:{}".format(C,H,W))
        self.W_s = nn.Parameter(torch.randn(self.C,K)).cuda()
        self.W_i = nn.Parameter(torch.randn(K,1)).cuda()
        self.bs = nn.Parameter(torch.randn(K)).cuda()
        self.bi = nn.Parameter(torch.randn(1)).cuda()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 0)  # softmax layer to calculate weights
        
    def forward(self, feature_map):
        """
        Forward propagation.
        :param feature_map: feature map in level L(batch_size, C, H, W)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: alpha
        """
        V_map = feature_map.view(feature_map.shape[0],self.C,-1) 
        V_map = V_map.permute(0,2,1)#(batch_size,W*H,C)
        # print(V_map.shape)
        # print("m1",torch.matmul(V_map,self.W_s).shape)
        # print("m2",torch.matmul(decoder_hidden,self.W_hs).shape)
        att = self.tanh((torch.matmul(V_map,self.W_s)+self.bs))# + (torch.matmul(decoder_hidden,self.W_hs).unsqueeze(1)))#(batch_size,W*H,C)
        # print("att",att.shape)
        alpha = self.softmax(torch.matmul(att,self.W_i) + self.bi)
#         print("alpha",alpha.shape)
        alpha = alpha.squeeze(2)
        feature_map = feature_map.view(feature_map.shape[0],self.C,-1) 
        # print("feature_map",feature_map.shape)
        # print("alpha",alpha.shape)
        temp_alpha = alpha.unsqueeze(1)
        attention_weighted_encoding = torch.mul(feature_map,temp_alpha).view(feature_map.shape[0],feature_map.shape[1],self.W,self.H)
        return attention_weighted_encoding,alpha



class model_fusion(nn.Module):
    def __init__(self,sigma=0.5,alpha=0.5):
        super(model_fusion, self).__init__()
        # self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model = densenet.densenet161(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.classifier = nn.Linear(2208, 271)
        self.sigma = sigma
        self.alpha = alpha
        self.conv1 = conv3x3(3648,271)
        self.conv2 = conv3x3(271,271)
        self.bn = nn.BatchNorm2d(271)
        self.classifier = nn.Linear(271, 271)


    def forward(self, x):
        model_out, feature1, feature2, feature3 = self.pretrained_model(x)

        # print(model_out.shape)
        # print(feature1.shape)
        # print(feature2.shape)
        # print(feature3.shape)

        feature1_att, _ = Spatial_attention(feature1)(feature1)
        feature1_att = F.interpolate(feature1_att, size=(feature1.shape[2],feature1.shape[3]), mode='bilinear')

        feature2_att, _ = Spatial_attention(feature2)(feature2)
        feature2_att = F.interpolate(feature2_att, size=(feature1.shape[2],feature1.shape[3]), mode='bilinear')

        feature3_att, _ = Spatial_attention(feature3)(feature3)
        feature3_att = F.interpolate(feature3_att, size=(feature1.shape[2],feature1.shape[3]), mode='bilinear')

        feature_att_map = torch.cat((feature1_att,feature2_att,feature3_att),dim=1)
        feature_att_mean = torch.mean(feature_att_map,dim=1)

        mask = torch.gt(feature_att_map, feature_att_mean.unsqueeze(1))

        feature_att_filter_map = torch.where(mask,torch.full_like(feature_att_map, 0.),feature_att_map)

        feature2 = F.interpolate(feature2, size=(feature1.shape[2],feature1.shape[3]), mode='bilinear')
        feature3 = F.interpolate(feature3, size=(feature1.shape[2],feature1.shape[3]), mode='bilinear')

        feature_map = torch.cat((feature1,feature2,feature3),dim=1)

        out = feature_map + self.sigma * feature_att_filter_map

        out = self.conv1(out)
        out = self.bn(out)
        out = self.conv2(out)
        out = self.pretrained_model.avgpool(out)
        out = out.view(x.shape[0],-1)
        out = model_out + self.alpha * self.classifier(out)


        return out 

if __name__ == '__main__':
    # model = densenet.densenet161(pretrained=True)
    # model.avgpool = nn.AdaptiveAvgPool2d(1)
    feature_in = torch.randn([4,3,224,224])
    # out, f1, f2, f3 = model(feature_in)
    # print(out.shape)
    # print(f1.shape)
    # print(f2.shape)
    # print(f3.shape)
    # for name, param in model.named_parameters():
	#     print(name, '      ', param.size())
    # feature_map = torch.randn([4,3,224,224])
    model = model_fusion()

    out = model(feature_in)
    print(out.shape)




