import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(
        self,
        n_input_channels: int = 1,
        n_output_channels: int = 256,
        p_dropout: float = 0.0,
        time_downsample_ratio: int = 16,
        **kwargs
    ):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.p_dropout = p_dropout
        self.n_output_channels = n_output_channels
        self.time_downsample_ratio = time_downsample_ratio

class Newmodel(BaseEncoder):
    def __init__(self, n_input_channels: int = 7, p_dropout: float = 0.0):
        super(Newmodel, self).__init__(
            n_input_channels=n_input_channels,
            n_output_channels=256,
            p_dropout=p_dropout,
            time_downsample_ratio=16,
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.99),
            nn.GELU(),
            nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.99),
            nn.GELU()
        )

        self.conv1 = GLFE(inplanes=32, planes=64)
        self.conv2 = GLFE(inplanes=64, planes=128)
        self.conv3 = GLFE(inplanes=128, planes=256)


        self.pool1 = nn.AvgPool2d([2, 2])
        self.drop1 = nn.Dropout(p_dropout)
        self.drop2 = nn.Dropout(p_dropout)
        self.drop3 = nn.Dropout(p_dropout)
        self.drop4 = nn.Dropout2d(0.2)

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = self.drop1(self.pool1(x1))
        x2 = self.drop2(self.pool1(self.conv1(x1)))
        x3 = self.drop3(self.pool1(self.conv2(x2)))
        x4 = self.drop4(self.pool1(self.conv3(x3)))

        return x4

class GLFE(nn.Module):
    def __init__(self, inplanes,planes):
        super(GLFE, self).__init__()
        self.pre = ODConv2d(inplanes, planes, 3, 3, kernel_num=1, reduction=0.25, padding=(1, 1))
        # omni-directional dynamic convolution
        self.epsa = MSFEBlock(planes, planes)
        #
        self.conv1x3 = nn.Conv2d(inplanes, planes, (1, 3), padding=(0, 1))
        self.conv3x1 = nn.Conv2d(inplanes, planes, (3, 1), padding=(1, 0))
        self.convh1x3 = nn.Conv2d(planes, planes, (1, 3), padding=(0, 1))
        self.convh3x1 = nn.Conv2d(planes, planes, (3, 1), padding=(1, 0))
        self.Conv3x1 = nn.Conv2d(planes, planes, (3, 1), padding=(1, 0))
        self.Conv1x3 = nn.Conv2d(planes, planes, (1, 3), padding=(0, 1))
        self.Convh3x1 = nn.Conv2d(planes, planes, (3, 1), padding=(1, 0))
        self.Convh1x3 = nn.Conv2d(planes, planes, (1, 3), padding=(0, 1))
        # #
        self.shortcuth = nn.Sequential(
            nn.Conv2d(inplanes, planes, (1, 1), bias=False),
            nn.BatchNorm2d(planes, eps=0.001, momentum=0.99),
        )


        self.ta = TriAttention()
        self.bn = nn.BatchNorm2d(planes, eps=0.001, momentum=0.99)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.99)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.99)

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.act3 = nn.GELU()
        self.act4 = nn.GELU()
        self.sf = AFF(planes, r=4)
        #Attentional feature fusion, proposed by"Attentional Feature Fusion-2021 WACV"


    def forward(self, x):

        out = self.act1(self.bn(self.pre(x)))
        out1 = self.epsa(out)
        #
        out1x3 = self.conv1x3(x)
        out3x1 = self.conv3x1(x)
        outh3x1 = self.convh3x1(out1x3)
        outh1x3 = self.convh1x3(out3x1)
        outm = self.act2(self.bn1(outh3x1 + outh1x3))
        x13 = self.Conv1x3(outm)
        x31 = self.Conv3x1(outm)
        out31 = self.Convh3x1(x13)
        out13 = self.Convh1x3(x31)
        out2 = self.act3(self.bn2(out13 + out31))


        out = self.sf(out1, out2)
        identity = self.shortcuth(x)
        out = self.act4(out + identity)
        out = self.ta(out)

        return out

class MSFEBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[1, 3, 5, 7],
                 conv_groups=[1, 2, 4, 8]):
        super(MSFEBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn1 = norm_layer(planes)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.act2(out+identity)

        return out


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[1, 3, 5, 7], stride=1, conv_groups=[1, 2, 4, 8]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)
        self.agger = nn.Sequential(
            nn.Conv2d(2*planes, planes, (1, 1), bias=False),
            nn.BatchNorm2d(planes),
            nn.GELU(),
            nn.Conv2d(planes, planes, (1, 1), bias=False)
        )
        self.init_weight()



    def forward(self, x):
        identity = x
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        identity1 = out
        outc = shuffle_channels(out, 4)
        outc = torch.cat((outc, out), dim=1)
        out1 = self.agger(outc)
        out = out1 + identity1 + identity
        return out

def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

class NPool(nn.Module):
    def __init__(self,k):
        super(NPool, self).__init__()
        self.k = k
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1d1 = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k//2)
        self.sig = nn.Sigmoid()


    def forward(self,x):
        avg = self.avg(x)
        out = avg.squeeze(-1).permute(0, 2, 1)
        out = self.conv1d1(out).permute(0, 2, 1).unsqueeze(-1)
        a = self.sig(out)
        out = a * x
        return out

class TriAttention(nn.Module):
    # Feature recalibration module consists of three attentional branches along channel, time and frequency dimensions
    def __init__(self):
        super(TriAttention, self).__init__()
        self.pool1 = NPool(3)
        self.pool2 = NPool(3)
        self.pool3 = NPool(3)


    def forward(self, x):

        x_perm1 = x.permute(0,2,1,3).contiguous() #B,T,C,F   T-branch
        x_out1 = self.pool1(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous() #B,F,T,C   F-branch
        x_out2 = self.pool2(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        x_out31 = self.pool3(x)   #C-branch
        x_out = 1/3 * (x_out31 + x_out11 + x_out21)

        return x_out


class SeldDecoder(nn.Module):
    """
    Decoder for SELD.
    input: batch_size x n_frames x input_size
    """
    def __init__(self, n_output_channels, n_classes: int = 12, output_format: str = 'reg_xyz', p_dropout: float = 0.1,
                 decoder_type: str = 'bigru', freq_pool: str = 'avg', decoder_size: int = 128, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.decoder_type = decoder_type
        self.freq_pool = freq_pool
        self.doa_format = output_format
        self.p_dropout = p_dropout


        if self.decoder_type == 'gru':
            self.gru_input_size = n_output_channels
            self.gru_size = decoder_size
            self.fc_size = self.gru_size

            self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=self.gru_size,
                              num_layers=2, batch_first=True, bidirectional=False, dropout=0.3)

        elif self.decoder_type == 'bigru':
            self.gru_input_size = n_output_channels
            # self.gru_input_size = 128
            self.gru_size = decoder_size
            self.fc_size = self.gru_size * 2
            self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=self.gru_size,
                              num_layers=2, batch_first=True, bidirectional=True, dropout=0.0)

        elif self.decoder_type == 'lstm':
            self.lstm_input_size = n_output_channels
            self.lstm_size = decoder_size
            self.fc_size = self.lstm_size

            self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.gru_size,
                                num_layers=2, batch_first=True, bidirectional=False, dropout=0.3)

        elif self.decoder_type == 'bilstm':
            self.lstm_input_size = n_output_channels
            self.lstm_size = decoder_size
            self.fc_size = self.lstm_size * 2

            self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.gru_size,
                               num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)


        # sed
        self.event_fc_1 = nn.Linear(self.fc_size, self.fc_size // 2, bias=True)
        self.event_dropout_1 = nn.Dropout(p=p_dropout)
        self.event_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.event_dropout_2 = nn.Dropout(p=p_dropout)


        # doa
        self.x_fc_1 = nn.Linear(self.fc_size, self.fc_size//2, bias=True)
        self.y_fc_1 = nn.Linear(self.fc_size, self.fc_size//2, bias=True)
        self.z_fc_1 = nn.Linear(self.fc_size, self.fc_size//2, bias=True)
        self.x_dropout_1 = nn.Dropout(p=p_dropout)
        self.y_dropout_1 = nn.Dropout(p=p_dropout)
        self.z_dropout_1 = nn.Dropout(p=p_dropout)
        self.x_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.y_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.z_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.x_dropout_2 = nn.Dropout(p=p_dropout)
        self.y_dropout_2 = nn.Dropout(p=p_dropout)
        self.z_dropout_2 = nn.Dropout(p=p_dropout)

        self.init_weights()

    def forward(self, x):
        """
        :params x: (batch_size, n_channels, n_timesteps/n_frames (downsampled), n_features/n_freqs (downsampled)
        """


        if self.freq_pool == 'avg':
            x = torch.mean(x, dim=3)
        elif self.freq_pool == 'max':
            (x, _) = torch.max(x, dim=3)
        elif self.freq_pool == 'avg_max':
            x1 = torch.mean(x, dim=3)
            (x, _) = torch.max(x, dim=3)
            x = x1 + x
        else:
            raise NotImplementedError('freq pooling {} is not implemented'.format(self.freq_pool))
        '''(batch_size, feature_maps, time_steps)'''

        # swap dimension: batch_size, n_timesteps, n_channels/n_features
        x = x.transpose(1, 2)



        if self.decoder_type in ['gru', 'bigru']:
            x, _ = self.gru(x)
        elif self.decoder_type in ['lsmt', 'bilstm']:
            x, _ = self.lstm(x)
        elif self.decoder_type == 'transformer':
            x = x.transpose(1, 2)  # undo swap: batch size,  n_features, n_timesteps,
            x = self.pe(x)  # batch_size, n_channels/n features, n_timesteps
            x = x.permute(2, 0, 1)  # T (n_timesteps), N (batch_size), C (n_features)
            x = self.decoder_layer(x)
            x = x.permute(1, 0, 2)  # batch_size, n_timesteps, n_features
        elif self.decoder_type == 'conformer':
            x = self.decoder_layer(x)
        if self.decoder_type =='moganet':
            x = self.decoder_layer(x)

        # SED: multi-label multi-class classification, without sigmoid
        event_frame_logit = F.relu_(self.event_fc_1(self.event_dropout_1(x)))  # (batch_size, time_steps, n_classes)
        event_frame_logit = self.event_fc_2(self.event_dropout_2(event_frame_logit))
        # event_frame_logit = torch.sigmoid((event_frame_logit))
        # DOA: regression
        x_output = F.relu_(self.x_fc_1(self.x_dropout_1(x)))
        x_output = torch.tanh(self.x_fc_2(self.x_dropout_2(x_output)))
        y_output = F.relu_(self.y_fc_1(self.y_dropout_1(x)))
        y_output = torch.tanh(self.y_fc_2(self.y_dropout_2(y_output)))
        z_output = F.relu_(self.z_fc_1(self.z_dropout_1(x)))
        z_output = torch.tanh(self.z_fc_2(self.z_dropout_2(z_output)))
        doa_output = torch.cat((x_output, y_output, z_output), dim=-1)  # (batch_size, time_steps, 3 * n_classes)
        output = {
            'event_frame_logit': event_frame_logit,
            'doa_frame_output': doa_output,
        }

        return output