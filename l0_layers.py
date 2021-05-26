import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init
import numpy as np
from scipy.stats import norm

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6


class L0Dense(Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""
    def __init__(self, in_features, out_features, bias=True, weight_decay=1., droprate_init=0.5, temperature=2./3.,
                 lamba=1., local_rep=False,P=10,la=0.1, **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Dense, self).__init__()
        self.P = P
        self.in_features = in_features
        self.out_features = out_features
        self.prior_prec = weight_decay
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.qz_loga = Parameter(torch.Tensor(in_features))
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.use_bias = False
        self.phi = torch.Tensor(in_features)
        self.u = torch.Tensor(in_features)
        self.local_rep = local_rep
        self.la = la
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weight, mode='fan_out')
        self.phi.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)
        self.u.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def reset_qz_loga(self):
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

    def reset_to_one(self):
        self.qz_loga.data.normal_(10,0.0)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    # def _reg_w(self):
    #     """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
    #     logpw_col = torch.sum(- (.5 * self.prior_prec * self.weight.pow(2)) - self.lamba, 1)
    #     logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
    #     logpb = 0 if not self.use_bias else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
    #     return logpw + logpb

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw = torch.sum(1 - q0)
        return logpw

    def myround(self,Z,T):
        Z = Z.squeeze()
        for i,z in enumerate(Z):
            if z<T:
                Z[i] = 0
            else:
                Z[i] = 1
        return Z

    def Encryption_Size(self):
        Z = self.sample_z(1)
        Z = self.myround(Z,0.9)
        Z = Z.detach().cpu().numpy()
        Z = Z.squeeze()
        if np.sum(Z)/Z.size>0.5:
            b = round(0.5*Z.size)
            SIZE = b
            return SIZE
        else:
            SIZE = Z.size
            return SIZE

    def ReconstructionError(self):
        J = torch.norm(self.qz_loga.cpu() - self.phi + self.u, 2).pow(2)
        return J
    def regularization(self):
        return self._reg_w()

    def constraint(self):
        return torch.sum((1 - self.cdf_qz(0)))

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.in_features).expand(batch_size, self.in_features)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.in_features)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1) * self.weight

    def forward(self, input):
        if not self.training:
            output = input.mm(self.weight)
            return output
        if self.local_rep or not self.training:
            z = self.sample_z(input.size(0), sample=self.training)
            xin = input.mul(z)
            output = xin.mm(self.weight)
        else:
            weights = self.sample_weights()
            output = input.mm(weights)
        if self.use_bias:
            output.add_(self.bias)
        return output

    def __repr__(self):
        s = ('{name}({in_features} -> {out_features}, droprate_init={droprate_init}, '
             'lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, '
             'local_rep={local_rep}')
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def Encrypt(self):
        Z = self.sample_z(1)
        Z = self.myround(Z,0.9)
        Z = Z.detach().cpu().numpy()
        SEED = np.random.randint(0,100000,size=Z.shape).squeeze()
        SIZE, location = self.weightEncrypt(Z,SEED)
        return SEED,location,SIZE

    def WeightsEncrypt(self,Z,SEED):
        SIZE = 0
        Z = Z.squeeze()
        if np.sum(Z)/Z.size>0.5:
            b = round(0.5*Z.size)
            a = np.argpartition(Z, b, axis=0)
            location = a[b:]
            for i in location:
                seed = SEED[i]
                torch.manual_seed(seed)
                weight = self.weight[0]
                a = np.random.random_sample(weight.shape)
                maxvalue = weight.detach().max().cpu().numpy()
                minvalue = weight.detach().min().cpu().numpy()
                ReplaceValue = (maxvalue - minvalue) * a + minvalue
                SIZE = SIZE + ReplaceValue.size
                self.weight[i] = torch.from_numpy(np.array(ReplaceValue))
            return SIZE,location
        else:
            location = []
            for i,z in enumerate(Z):
                if not z == 0:
                    location.append(i)
                    seed = SEED[i]
                    torch.manual_seed(seed)
                    weight = self.weight[0]
                    a = np.random.random_sample(weight.shape)
                    maxvalue = weight.detach().max().cpu().numpy()
                    minvalue = weight.detach().min().cpu().numpy()
                    ReplaceValue = (maxvalue - minvalue) * a + minvalue
                    SIZE = SIZE + ReplaceValue.size
                    self.weight[i] = torch.from_numpy(np.array(ReplaceValue))
            return SIZE,np.array(location)

    def Random_Encryption(self):
        Z = self.sample_z(1)
        Z = self.myround(Z, 0.9)
        Z = Z.detach().cpu().numpy()
        Z = Z.squeeze()
        maxvalue = self.weight.detach().max().cpu().numpy()
        minvalue = self.weight.detach().min().cpu().numpy()
        if np.sum(Z) / Z.size > 0.5:
            b = round(0.5 * Z.size)
            SIZE = b
            Index = np.random.permutation(self.qz_loga.shape[0])[:b]
            for i in Index:
                a = np.random.random_sample(self.weight[i].shape)
                ReplaceValue = (maxvalue - minvalue) * a + minvalue
                self.weight[i] = torch.from_numpy(np.array(ReplaceValue))
            return SIZE
        else:
            SIZE = np.sum(Z)
            Index = np.random.permutation(self.qz_loga.shape[0])[:int(SIZE)]
            for i in Index:
                a = np.random.random_sample(self.weight[i].shape)
                ReplaceValue = (maxvalue - minvalue) * a + minvalue
                self.weight[i] = torch.from_numpy(np.array(ReplaceValue))
            return SIZE


class L0Conv2d(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 droprate_init=0.5, temperature=2./3., weight_decay=1., lamba=1., local_rep=False,P=10,la=0.1,
                 SEED=0,Ratio = 0,**kwargs):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param weight_decay: Strength of the L2 penalty
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.P = P
        self.la = la
        self.SEED = SEED
        self.Ratio = Ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.prior_prec = weight_decay
        self.lamba = lamba
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.temperature = temperature
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.use_bias = False
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.phi = torch.Tensor(out_channels*in_channels)
        self.u = torch.Tensor(out_channels*in_channels)
        self.qz_loga = Parameter(torch.Tensor(out_channels*in_channels))
        self.dim_z = out_channels*in_channels
        self.input_shape = None
        self.local_rep = local_rep

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.use_bias = True

        self.reset_parameters()
        self.reset_to_10()
        # print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weight, mode='fan_in')
        self.phi.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)
        self.u.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)
        self.weight.requires_grad = True
        if self.use_bias:
            self.bias.requires_grad = True
        if self.use_bias:
            self.bias.data.fill_(0)

    def reset_qz_loga(self):
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 4)

    def reset_to_zeros(self):
        self.qz_loga.data.normal_(0, 0.00001)

    def reset(self,mu):
        self.qz_loga.data.normal_(mu, 0.00001)

    def ReconstructionError(self):
        J = torch.norm(self.qz_loga.cpu()-self.phi+self.u,2).pow(2)
        return J

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw = torch.sum(1 - q0)
        return logpw

    def ToFClocation(self,EncryptedPlace):
        EncryptedPlace = torch.unique(EncryptedPlace)
        EncryptedPlace = torch.sort(EncryptedPlace,dim=0)[0]
        EncryptedPlace = EncryptedPlace.cpu()
        EncryptedPlace = EncryptedPlace.float()
        EncryptedPlace = torch.Tensor(EncryptedPlace)
        EncryptedPlace = EncryptedPlace.squeeze()
        FClocation = torch.zeros([EncryptedPlace.shape[0],2])
        NF = self.weight.shape[0]
        NC = self.weight.shape[1]
        FClocation[:,0] = torch.Tensor(np.floor(EncryptedPlace/NC))
        FClocation[:,1] = torch.Tensor(np.mod(EncryptedPlace,NC))
        return FClocation

    # def Encrypt(self,R,PlaceMode,ValueMode):
    #     EncryptedSize = int(np.floor(R * self.qz_loga.shape[0]) + 1)
    #     if PlaceMode == 'Selection':
    #         EncryptedPlace = EncryptedPlace.squeeze()
    #         EncryptedPlace = torch.round(EncryptedPlace - 0.12)
    #         EncryptedPlace = torch.nonzero(EncryptedPlace > 0)
    #         FClocation = self.ToFClocation(EncryptedPlace)
    #         for location in FClocation:
    #             Fi = int(location[0].numpy())
    #             Ci = int(location[1].numpy())
    #             ReplaceWeight = self.weight[Fi,Ci]
    #             RandomValue = self.EncryptValue(ReplaceWeight, ValueMode)
    #             ReplaceValue = self.weight[Fi,Ci].Jpeg.cpu().numpy() + RandomValue
    #             # ReplaceValue = ReplaceValue.astype(np.float32)
    #             self.weight[Fi,Ci] = torch.from_numpy(np.array(ReplaceValue)).cuda()
    #     elif PlaceMode == 'Random':
    #         # EncryptedPlace = EncryptedPlace.squeeze()
    #         # EncryptedPlace = torch.round(EncryptedPlace - 0.12)
    #         # EncryptedPlace = torch.nonzero(EncryptedPlace > 0)
    #         Index = np.random.permutation(self.qz_loga.shape[0])[:EncryptedSize]
    #         Index = torch.Tensor(Index).float()
    #         FClocation = self.ToFClocation(Index)
    #         for location in FClocation:
    #                 Fi = int(location[0].numpy())
    #                 Ci = int(location[1].numpy())
    #                 ReplaceWeight = self.weight[Fi,Ci]
    #                 RandomValue = self.EncryptValue(ReplaceWeight,ValueMode)
    #                 ReplaceValue = self.weight[Fi,Ci].Jpeg.cpu().numpy() + RandomValue
    #                 self.weight[Fi,Ci] = torch.from_numpy(np.array(ReplaceValue)).cuda()
    #     elif PlaceMode == 'TopK':
    #         Jpeg = self.weight.Jpeg.detach().cpu().numpy()
    #         mean = np.mean(Jpeg.reshape(Jpeg.shape[0],Jpeg.shape[1],9),axis=2)
    #         mean = mean.reshape(Jpeg.shape[0]*Jpeg.shape[1])
    #         meanIndex = np.argsort(-abs(mean),kind='mergesort')[:EncryptedSize]
    #         meanIndex = torch.Tensor(meanIndex)
    #         FClocation = self.ToFClocation(meanIndex)
    #         for location in FClocation:
    #                 Fi = int(location[0].numpy())
    #                 Ci = int(location[1].numpy())
    #                 ReplaceWeight = self.weight[Fi,Ci]
    #                 RandomValue = self.EncryptValue(ReplaceWeight,ValueMode)
    #                 ReplaceValue = self.weight[Fi,Ci].Jpeg.cpu().numpy() + RandomValue
    #                 self.weight[Fi,Ci] = torch.from_numpy(np.array(ReplaceValue)).cuda()
    #     elif PlaceMode == 'LastK':
    #         Jpeg = self.weight.Jpeg.detach().cpu().numpy()
    #         mean = np.mean(Jpeg.reshape(Jpeg.shape[0],Jpeg.shape[1],9),axis=2)
    #         mean = mean.reshape(Jpeg.shape[0]*Jpeg.shape[1])
    #         meanIndex = np.argsort(abs(mean),kind='mergesort')[:EncryptedSize]
    #         meanIndex = torch.Tensor(meanIndex)
    #         FClocation = self.ToFClocation(meanIndex)
    #         for location in FClocation:
    #                 Fi = int(location[0].numpy())
    #                 Ci = int(location[1].numpy())
    #                 ReplaceWeight = self.weight[Fi,Ci]
    #                 RandomValue = self.EncryptValue(ReplaceWeight,ValueMode)
    #                 ReplaceValue = self.weight[Fi,Ci].Jpeg.cpu().numpy() + RandomValue
    #                 self.weight[Fi,Ci] = torch.from_numpy(np.array(ReplaceValue)).cuda()
    #     return FClocation

    def EncryptValue(self,ReplaceWeight,mode,seed):
        if mode == 'Random':
            maxvalue = self.weight.detach().max().cpu().numpy()
            minvalue = self.weight.detach().min().cpu().numpy()
            a = np.random.random_sample(ReplaceWeight.shape)
            RANDOMVALUE = (maxvalue - minvalue) * a + minvalue
        elif mode == 'Gaussian':
            data = self.weight.data.detach().cpu().numpy()
            mean = np.mean(data.reshape(-1,3,3),axis=0)
            # mean = np.mean(Jpeg.reshape(-1), axis=0)
            mean = mean-ReplaceWeight.data.detach().cpu().numpy()
            mean = mean.reshape(-1, 1)
            std = np.std(data.reshape(-1,3,3),axis=0).reshape(-1,1)
            # std = np.std(Jpeg.reshape(-1), axis=0).reshape(-1, 1)
            # std = torch.Tensor(std)
            # RANDOMVALUE = torch.zeros(ReplaceWeight.view(-1,1).shape)
            RANDOMVALUE = np.zeros(ReplaceWeight.view(-1, 1).shape)
            np.random.seed(seed)
            RANDOMVALUE = np.random.normal(mean,std)
            # [RANDOMVALUE[i].random.normal(float(mean[i]),float(std[i])) for i in range(0,len(mean))]
            # [RANDOMVALUE[i].Jpeg.normal_(float(mean[i]), float(std)) for i in range(0,9)]
        return RANDOMVALUE.reshape(self.weight.shape[2],self.weight.shape[3])

    def EncryptValueGroup(self,ReplaceWeight,mode,seed):
        if mode == 'Random':
            maxvalue = self.weight.detach().max().cpu().numpy()
            minvalue = self.weight.detach().min().cpu().numpy()
            a = np.random.random_sample(ReplaceWeight.shape)
            RANDOMVALUE = (maxvalue - minvalue) * a + minvalue
        elif mode == 'Gaussian':
            data = self.weight.data.detach().cpu().numpy()
            mean = np.mean(data.reshape(-1,3,3),axis=0)
            mean = np.repeat(mean.reshape(1,3,3),ReplaceWeight.shape[0],axis=0)\
                   -ReplaceWeight.data.detach().cpu().numpy()
            mean = mean.reshape(-1, 1)
            std = np.std(data.reshape(-1,3,3),axis=0)
            std = np.repeat(std.reshape(1, 3, 3), ReplaceWeight.shape[0], axis=0)
            std = std.reshape(-1,1)
            np.random.seed(seed)
            RANDOMVALUE = np.random.normal(mean,std)
        return RANDOMVALUE.reshape(ReplaceWeight.shape), std, mean

    def DPE(self,DL,seed):
        F = DL[:, 0].tolist()
        C = DL[:, 1].tolist()
        CT = self.Encryption(DL,seed)
        CT = self.Mapping(CT)
        self.weight.requires_grad = False
        self.weight[F, C, :] = torch.from_numpy(np.array(CT.astype(np.float32))).cuda()
        return CT

    def Encryption(self,DL,seed):
        ## DL: Dominated Locations
        F = DL[:, 0].tolist()
        C = DL[:, 1].tolist()
        SelectedWeight = self.weight[F,C,:,:]
        np.random.seed(seed)
        a = np.random.random_sample(SelectedWeight.shape)
        CipherText = SelectedWeight.cpu() + torch.from_numpy(a.astype(np.float32))
        return CipherText

    def Mapping(self,CT):
        shape = CT.shape
        ## CT: CiphereText
        CT = CT.detach().cpu().numpy().reshape(-1,1,1,1).squeeze()
        data = self.weight.data.detach().cpu().numpy()
        mu = np.mean(data.reshape(-1, 1, 1, 1), axis=0)
        std = np.std(data.reshape(-1, 1, 1, 1), axis=0)
        h = np.min(CT)-(1e-6)
        f = np.max(CT)+(1e-6)
        CT = (CT-h)/(f-h)
        CT = norm.ppf(CT, loc=mu, scale=std)
        CT = CT.reshape(shape)
        return CT

    def EncryptLocation(self,Ratio,PlaceMode,M=False):
        Size = np.floor(self.qz_loga.size()[0]*Ratio)+1
        Size = int(Size)
        if Size > self.qz_loga.size()[0]:
            Size = self.qz_loga.size()[0]
        if PlaceMode == 'PSS':
            ES = []
            if M:
                for m in range(M):
                    a = int(np.floor((0.6+((1-0.6)*(m+1)/M)) * Size))
                    EncryptedPlace = torch.sort(-self.qz_loga, dim=0)[1][0:a]
                    FClocation = self.ToFClocation(EncryptedPlace)
                    ES.append(FClocation)
                return ES
            else:
                EncryptedPlace = torch.sort(-self.qz_loga, dim=0)[1][:Size]
                FClocation = self.ToFClocation(EncryptedPlace)

        elif PlaceMode == 'Random':
            Index = np.random.permutation(self.qz_loga.shape[0])[:Size]
            Index = torch.Tensor(Index).float()
            FClocation = self.ToFClocation(Index)
        elif PlaceMode == 'Descend':
            data = self.weight.data.detach().cpu().numpy()
            mean = np.mean(data.reshape(data.shape[0],data.shape[1],9),axis=2)
            mean = mean.reshape(data.shape[0]*data.shape[1])
            meanIndex = np.argsort(-abs(mean),kind='mergesort')[:Size]
            meanIndex = torch.Tensor(meanIndex)
            FClocation = self.ToFClocation(meanIndex)
        elif PlaceMode == 'Ascend':
            data = self.weight.data.detach().cpu().numpy()
            mean = np.mean(data.reshape(data.shape[0],data.shape[1],9),axis=2)
            mean = mean.reshape(data.shape[0]*data.shape[1])
            meanIndex = np.argsort(abs(mean),kind='mergesort')[:Size]
            meanIndex = torch.Tensor(meanIndex)
            FClocation = self.ToFClocation(meanIndex)
        elif PlaceMode == 'Mean':
            data = self.weight.data.detach().cpu().numpy()
            mean = np.mean(data.reshape(data.shape[0],data.shape[1],9),axis=2)
            mean = mean.reshape(data.shape[0]*data.shape[1])
            mean = abs(abs(mean)-np.mean(abs(mean)))
            meanIndex = np.argsort(mean, kind='mergesort')[:Size]
            meanIndex = torch.Tensor(meanIndex)
            FClocation = self.ToFClocation(meanIndex)
        return FClocation

    def Encrypt(self,FClocation,ValueMode,seed):
        for location in FClocation:
                Fi = int(location[0].numpy())
                Ci = int(location[1].numpy())
                ReplaceWeight = self.weight[Fi,Ci]
                RandomValue = self.EncryptValue(ReplaceWeight,ValueMode,seed)
                ReplaceValue = self.weight[Fi,Ci].data.cpu().numpy() + RandomValue
                self.weight[Fi,Ci] = torch.from_numpy(np.array(ReplaceValue)).cuda()
        return FClocation

    def DPRM(self,FClocation,ValueMode,seed):
        if FClocation == []:
            return
        F = FClocation[:, 0].tolist()
        C = FClocation[:, 1].tolist()
        ReplaceWeightGroup = self.weight[F,C,:,:]
        RandomValueGroup,std, mean = self.EncryptValueGroup(ReplaceWeightGroup, ValueMode,seed)
        ori = self.weight.data.cpu().numpy()
        ReplaceValueGroup = self.weight[F, C,:].data.cpu().numpy() + RandomValueGroup
        self.weight.requires_grad = False
        self.weight[F,C,:] = torch.from_numpy(np.array(ReplaceValueGroup.astype(np.float32)))
        return ReplaceValueGroup,RandomValueGroup,std, mean,ori

    def De_DPRM(self,FClocation,seed,std,mean,m):
        TEMP = torch.zeros(self.weight.shape).float()
        if FClocation == []:
            return
        FT = FClocation[len(FClocation)-1][:, 0].tolist()
        CT = FClocation[len(FClocation)-1][:, 1].tolist()
        F = FClocation[m][:, 0].tolist()
        C = FClocation[m][:, 1].tolist()
        np.random.seed(seed)
        RANDOMVALUE = np.random.normal(mean, std)
        RANDOMVALUE = RANDOMVALUE.reshape(self.weight[FT, CT,:,:].shape)
        ReplaceValueGroup = self.weight[FT, CT, :].data.cpu().numpy() - RANDOMVALUE
        TEMP[FT, CT, :] = torch.from_numpy(ReplaceValueGroup.astype(np.float32))
        self.weight.requires_grad = False
        self.weight[F,C,:] = TEMP[F,C,:]
        return self.weight,self.weight

    def DPRM_Loss(self,FClocation,ValueMode,seed):
        Noise = torch.zeros(self.weight.shape)
        if FClocation == []:
            return
        F = FClocation[:, 0].tolist()
        C = FClocation[:, 1].tolist()
        ReplaceWeightGroup = self.weight[F,C,:,:]
        RandomValueGroup = self.EncryptValueGroup(ReplaceWeightGroup, ValueMode,seed)
        Noise[F,C,:] = torch.from_numpy(RandomValueGroup).float()
        return Noise

    def myround(self,Z,T):
        Z = Z.squeeze()
        for i,z in enumerate(Z):
            if z<T:
                Z[i] = 0
            else:
                Z[i] = 1
        return Z

    def regularization(self):
        return self._reg_w()

    def WeightEncrypt(self,Z,SEED,ChannelSize):
        SIZE = 0
        Z = Z.squeeze()
        location = []
        for i,z in enumerate(Z):
            if not z == 0:
                ChannelIndex = np.random.permutation(self.weight.shape[1])[:ChannelSize]
                location.append(i)
                seed = SEED[i]
                torch.manual_seed(seed)
                weight = self.weight[0]
                maxvalue = weight.detach().max().cpu().numpy()
                minvalue = weight.detach().min().cpu().numpy()
                a = np.random.random_sample(self.weight[i, ChannelIndex].shape)
                RandomValue = (maxvalue - minvalue) * a + minvalue
                ReplaceValue = self.weight[i, ChannelIndex].data.cpu().numpy() + RandomValue
                SIZE = SIZE + ReplaceValue.size
                self.weight[i,ChannelIndex] = torch.from_numpy(np.array(ReplaceValue)).float().cuda()
        return SIZE,np.array(location)

    def constraint(self):
        return torch.sum((1 - self.cdf_qz(0)))

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def reset_to_one(self):
        self.qz_loga.data.normal_(4.2,0.00001)

    def reset_to_10(self):
        self.qz_loga.data.normal_(100.0,0.01)

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.dim_z))
            z = self.quantile_concrete(eps).view(batch_size, self.dim_z, 1, 1)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
        return F.hardtanh(z, min_val=0, max_val=1) * self.weight

    def ReshapeToFC(self,index):
        FCIndex = np.zeros([index.size,2])
        NF = self.weight.shape[0]
        NC = self.weight.shape[1]
        Findex = np.floor(index/NF)
        Cindex = index-((Findex)*NC)
        FCIndex[:,0] = Findex
        FCIndex[:,1] = Cindex
        return  FCIndex

    # def Random_Encryption(self,EncryptedPlace,mode):
    #     if mode == 'f':
    #         EncryptedPlace = EncryptedPlace.squeeze()
    #         Index = np.random.permutation(self.qz_loga.shape[0])[:EncryptedPlace.shape[0]]
    #         FClocation = self.ToFClocation(Index)
    #         for location in Index:
    #             ChannelIndex = np.random.permutation(self.weight.shape[1])[:ChannelSize]
    #             CIndex.append(ChannelIndex)
    #             maxvalue = self.weight[i].detach().max().cpu().numpy()
    #             minvalue = self.weight[i].detach().min().cpu().numpy()
    #             a = np.random.random_sample(self.weight[i, ChannelIndex].shape)
    #             RandomValue = (maxvalue - minvalue) * a + minvalue
    #             ReplaceValue = self.weight[i,ChannelIndex].Jpeg.cpu().numpy() + RandomValue
    #             ReplaceValue = ReplaceValue.astype(np.float32)
    #             self.weight[i, ChannelIndex] = torch.from_numpy(np.array(ReplaceValue))
    #         return Index,CIndex
    #     if mode == 'c':
    #         Index = np.random.permutation(self.weight.shape[0]*self.weight.shape[1])[:FilterSize*ChannelSize]
    #         FCIndex = self.ReshapeToFC(Index)
    #         for i in range(FCIndex.shape[0]):
    #             location = FCIndex[i,:]
    #             Fi = int(location[0])
    #             Ci = int(location[1])
    #             maxvalue = self.weight[Fi].detach().max().cpu().numpy()
    #             minvalue = self.weight[Fi].detach().min().cpu().numpy()
    #             a = np.random.random_sample(self.weight[Fi,Ci].shape)
    #             RandomValue = (maxvalue - minvalue) * a + minvalue
    #             ReplaceValue = self.weight[Fi,Ci].Jpeg.cpu().numpy() + RandomValue
    #             ReplaceValue = ReplaceValue.astype(np.float32)
    #             self.weight[Fi,Ci] = torch.from_numpy(np.array(ReplaceValue))
    #         return FCIndex,[]
    #
    #     # ChannelIndex = np.random.permutation(self.weight.shape[1])[:ChannelSize]
    #     # maxvalue = self.weight[FilterSize].detach().max().cpu().numpy()
    #     # minvalue = self.weight[FilterSize].detach().min().cpu().numpy()
    #     # a = np.random.random_sample(self.weight[FilterSize,ChannelIndex].shape)
    #     # RandomValue = (maxvalue - minvalue) * a + minvalue
    #     # ReplaceValue = self.weight[FilterSize,ChannelIndex].Jpeg.cpu().numpy() + RandomValue
    #     # ReplaceValue = ReplaceValue.astype(np.float32)
    #     # self.weight[FilterSize,ChannelIndex] = torch.from_numpy(np.array(ReplaceValue))
    #     # return FilterSize
    def vanishweight(self,Ratio):
        Size = np.floor(self.qz_loga.size()[0]*Ratio)+1
        Size = int(Size )
        EncryptedPlace = np.argpartition(-self.qz_loga.cpu().detach().numpy(), Size)[0:Size]
        EncryptedPlace = torch.Tensor(EncryptedPlace)
        FClocation = self.ToFClocation(EncryptedPlace)
        for location in FClocation:
            Fi = int(location[0].numpy())
            Ci = int(location[1].numpy())
            ReplaceValue = torch.zeros_like(self.weight[Fi, Ci])
            self.weight[Fi, Ci] = ReplaceValue.cuda()
    def forward(self, input_):
        z = self.sample_z(1, sample=self.training)
        Newdata = self.weight*z.view([self.weight.shape[0],self.weight.shape[1],1,1])

        # Location = self.EncryptLocation(self.Ratio,'PSS')
        # RandomValueGroup = self.DPRM_Loss(Location,'Gaussian',self.SEED)
        # Newdata = self.weight*z.view([self.weight.shape[0],self.weight.shape[1],1,1])+RandomValueGroup

        if self.use_bias:
            output = F.conv2d(input_, Newdata, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            output = F.conv2d(input_, Newdata, None, self.stride, self.padding, self.dilation, self.groups)
        return output

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, '
             'droprate_init={droprate_init}, temperature={temperature}, prior_prec={prior_prec}, '
             'lamba={lamba}, local_rep={local_rep}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)






