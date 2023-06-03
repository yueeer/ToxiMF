import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence


class dvib(nn.Module):
    def __init__(self, k, out_channels, hidden_size):
        super(dvib, self).__init__()

        # self.PNN = PNN(field_num=3, embedding_dim=256, output_dim_list=[512, 256])

        self.conv = torch.nn.Conv2d(in_channels=1,
                                    out_channels=out_channels,
                                    kernel_size=(1, 20),
                                    stride=(1, 1),
                                    padding=(0, 0),
                                    )

        self.rnn = torch.nn.GRU(input_size=out_channels,
                                hidden_size=hidden_size,
                                num_layers=2,
                                bidirectional=True,
                                batch_first=True,
                                dropout=0.2
                                )

        self.dec = nn.Linear(k, 2)

        self.f1 = nn.Linear(1280, 256)
        self.f2 = nn.Linear(100, 256)
        self.f3 = nn.Linear(hidden_size * 4, 256)

        self.output_layer_1 = nn.Linear(256 * 3, k)


    def cnn_gru(self, x, lens):
        '''

        :param x: torch.Size([100, 200, 20])
        :param lens: torch.Size([100])
        :return:
        '''
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.nn.ReLU()(x)
        x = x.squeeze(3)
        x = x.permute(0, 2, 1)

        gru_input = pack_padded_sequence(x, lens, batch_first=True)
        output, hidden = self.rnn(gru_input)
        output_all = torch.cat([hidden[-1], hidden[-2], hidden[-3], hidden[-4]], dim=1)
        return output_all

    def forward(self, pssm, lengths, fea, bertF,  device):
        cnn_vectors = self.cnn_gru(pssm, lengths)
        cnn_vectors1 = self.f3(cnn_vectors)  # torch.Size([100, 256])
        new_BertF = self.f1(bertF)
        fea1 = self.f2(fea)

        feature_vec = torch.cat([cnn_vectors1, new_BertF, fea1], dim=1)

        latent = self.output_layer_1(feature_vec)
        outputs = self.dec(latent)
        return outputs