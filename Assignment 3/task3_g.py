from torchvision import utils, models

tensor = models.resnet18(pretrained=True).conv1.weight.data
utils.save_image(tensor, 'data/filters.png', nrow=8)

