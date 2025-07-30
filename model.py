from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.inference.speaker import EncoderClassifier

import torch                    # Pytorch module 
import torch.nn as nn           # for creating  neural networks
from aasist import Model

class ECAPAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ECAPA_TDNN(input_size=80, lin_neurons=192, channels=[1024, 1024, 1024, 1024, 3072])

        pretrained = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        state_dict = pretrained.mods.embedding_model.state_dict()
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, features):  # features: [B, T, F]
        return self.model(features)  # output: x-vector [B, 192]

class AASISTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Model(config)
        state_dict = torch.load("weights/AASIST.pth", map_location="cpu")
        self.model.load_state_dict(state_dict)
        
    def forward(self, x):  # x: waveform [B, T]

        logits = self.model(x)
        return logits  #[B, 192]
    
class FusionModule(nn.Module):
    def __init__(self, input_dim=544, hidden_dim=256, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, fused_input):
        fused = self.mlp(fused_input)  # [B, 2]
        return fused
    
class SASVModel(nn.Module):
    def __init__(self, aasist_config):
        super().__init__()
        self.aasist = AASISTEncoder(aasist_config["model_config"])

        self.ecapa = ECAPAEncoder()
        self.fusion_layer = FusionModule()
        self.fc_out = torch.nn.Linear(64, 2, bias = False)


    def forward(self, tst_waveform, enr_features, tst_feature):
        # waveform: [B, T], features: [B, T, F]
        cm_emb, out = self.aasist(tst_waveform)     
        asv_enr_embed = self.ecapa(enr_features)    
        asv_tst_embed = self.ecapa(tst_feature) 

        asv_enr = torch.squeeze(asv_enr_embed) # shape: (bs, 192)
        asv_tst = torch.squeeze(asv_tst_embed) # shape: (bs, 192)
        cm_tst = torch.squeeze(cm_emb) # shape: (bs, 160)

        fused = self.fusion_layer(torch.cat([asv_enr, asv_tst, cm_tst], dim = 1))

        out = self.fc_out(fused)

        return out

if __name__ == "__main__":    
    config_dict = {
    "model_config": {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SASVModel(config_dict).to(device)

    # Dummy data
    batch_size = 5
    waveform = torch.randn(batch_size, 64600).to(device)       # mono waveform
    features = torch.randn(batch_size, 300, 80).to(device)     # log Mel-spectrogram
    print(features.shape)

    # Forward
    fused = model(waveform, features, features)

    print("Spoof score shape:", fused.shape)     # expect [B, 2] or [B, 1]

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print("Param count: ", pytorch_total_params) 
    print("ECAPA params:", sum(p.numel() for p in model.ecapa.parameters()))
    print("AASIST params:", sum(p.numel() for p in model.aasist.parameters()))
    print("Fusion params:", sum(p.numel() for p in model.fusion_layer.parameters()))
    print("Output FC params:", sum(p.numel() for p in model.fc_out.parameters()))
