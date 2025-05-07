import torch
import torch.nn as nn
import torch.nn.functional as F
from CNA import *


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1) 

    def forward(self, x):  
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0) 

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0) 

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0) 

class MaxNormDefaultConstraint(object):
    """
    Applies max L2 norm 2 to the weights until the final layer and L2 norm 0.5
    to the weights of the final layer as done in [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730

    """

    def apply(self, model):
        last_weight = None
        for name, module in list(model.named_children()):
            if hasattr(module, "weight") and (
                    not module.__class__.__name__.startswith("BatchNorm")
            ):
                module.weight.data = torch.renorm(
                    module.weight.data, 2, 0, maxnorm=2
                )
                last_weight = module.weight
        if last_weight is not None:
            last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)

class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [1, 2, 3, 5]
        cout1 = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout1, (1, kern), dilation=(1, dilation_factor)))
        self.norm = nn.BatchNorm2d(cout)

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class ListenNet(nn.Module):
    """
    ListenNet for the paper
    """
    def __init__(self, chans=64, samples=1000, num_classes=2, kernel=8, depth=16, avepool=10):
        super(ListenNet, self).__init__()
        self.channel_weight2 = torch.randn(16, 16).cuda().float()
        self.align = Align(chans, depth)
        
        self.temporal = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=depth, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(depth),
            nn.Conv2d(in_channels=depth, out_channels= depth, kernel_size=(1, kernel), groups=depth, bias=False),
            nn.BatchNorm2d(depth),
            nn.GELU(),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(depth, depth, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(depth),
            nn.Conv2d(depth, depth, kernel_size=(chans, 1), groups=depth, bias=False),
            nn.BatchNorm2d(depth),
            nn.GELU(),
        )
        
        self.MSTE = DilatedInception(depth, depth, dilation_factor= 1)
        self.BN = nn.BatchNorm2d(depth)
        self.skip_convs0 = nn.Conv2d(in_channels=chans, out_channels= depth, kernel_size=(1, 1))
        self.skip_convs1 = nn.Conv2d(in_channels= depth, out_channels= depth,kernel_size=(chans, 1), groups=depth,bias=False)
        self.dropout = 0.65
    

        self.skip0 = nn.Conv2d(in_channels= depth, 
                               out_channels= depth, 
                               kernel_size=(chans, 1), 
                               groups=depth,
                               bias=True)
  
        # 
        out = torch.ones((1, 1, chans, samples)) #1 1 64 128
        out = self.temporal(out) # 1 64 64 128
        out = self.spatial(out)
        N, C, H, W = out.size() # N=1 C=64 H=64 W=121 
        self.CNA = CNA(C, C//2)
        
        if W < avepool:
            avepool = W

        self.GAP = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        out = self.GAP(out)
        n_out_time = out.cpu().data.numpy().shape
        feat_dim = n_out_time[-1]*n_out_time[-2]*n_out_time[-3]
        self.classifier = nn.Linear(feat_dim , num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x): # 64,1,64,128
        # STDE
        Et = self.temporal(x)   # 64 64 64 128
        Es = self.spatial(Et)

        # MSTE
        InU = Et
        OutU = self.MSTE(InU)
        skip = self.skip0(F.dropout(OutU, self.dropout, training=self.training))
        skip_resized = F.interpolate(skip, size=(1, Es.size(-1)), mode='bilinear', align_corners=False)
        InEs = skip_resized + Es
        InEs = self.BN(InEs)

        # Align Reshape
        Et = Et.permute(0,2,1,3) 
        Et = self.align(Et)
        InEt = torch.einsum('bdcw,sc->bdsw', Et, self.channel_weight2)

        # CNA
        Est = self.CNA(InEt, InEs)

        # Classifer
        Est = self.GAP(Est)
        feat = torch.flatten(Est, 1)
        out = self.classifier(feat)
        
        return feat, out
    

