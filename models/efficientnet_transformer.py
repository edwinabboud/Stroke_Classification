import torch
import torch.nn as nn
import timm

class StrokeClassifier(nn.Module):
    def __init__(self, backbone='efficientnet_b0', pretrained=True, num_classes=3, dropout=0.3):
        super(StrokeClassifier, self).__init__()

        # Load pretrained EfficientNet from timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='avg')

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out


if __name__ == "__main__":
    model = StrokeClassifier()
    sample = torch.randn(2, 3, 224, 224)
    out = model(sample)
    print("Output shape:", out.shape)  # should be [2, 3]