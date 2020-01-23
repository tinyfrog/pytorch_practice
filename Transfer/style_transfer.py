import torchvision.models as models

resnet = models.resnet50(pretrained=True)

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.layer0 = nn.Sequential(*list(resnet.children())[0:1])
        self.layer1 = nn.Sequential(*list(resnet.children())[1:4])
        self.layer2 = nn.Sequential(*list(resnet.children())[4:5])
        self.layer3 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer4 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer5 = nn.Sequential(*list(resnet.children())[7:8])

    def forward(self, x):
        out_0 = self.layer0(x)
        out_1 = self.layer0(out_0)
        out_2 = self.layer0(out_1)
        out_3 = self.layer0(out_2)
        out_4 = self.layer0(out_3)
        out_5 = self.layer0(out_4)
        return out_0, out_1, out_2, out_3, out_4, out_5

for param in resnet.parameters():
    param.requires_grad = False

class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1, 2))
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return out