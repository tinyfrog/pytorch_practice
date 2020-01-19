import torchvision.models as models

resnet = models.resnet50(pretrained=True) # if pretrained false then initialize as random value

'''
for name, module in resnet.named_children():
    print(name)
'''

'''
layer0 : get pre-trained parameter
layer1 : create new model
'''

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.layer0 = nn.Sequential(*list(resnet.children())[0:-1])
        self.layer1 = nn.Sequential(
            nn.Linear(2048, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, num_category),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer0(x)
        out = out.view(batch_size, -1)
        out = self.layer1(out)
        return out

# we want to use pre-trained parameters so disable calculating slope
for params in model.layer0.parameters():
    params.require_grad = False

# Enable calculating slope
for params in model.layer1.parameters():
    params.requires_grad = True