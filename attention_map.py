#from inet import Model
import numpy as np
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from timm.models import create_model
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from torchvision import transforms

def to_tensor(img):
    transform_fn = Compose([Resize((224,224)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform_fn(img)

def show_img(img):
    img = np.asarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.imsave("./heatmap_png/heatmap.png", img)


def show_img2(img1, img2, alpha=0.3):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    plt.figure(figsize=(10, 10))
    plt.imshow(img1)
    plt.imshow(img2, alpha=alpha)
    plt.axis('off')
    plt.show()

def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 1:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward

img = Image.open('./cxr_chest/Lung_Opacity-1002.png').convert('RGB')
mask = Image.open('./cxr_chest/mask-Lung_Opacity-1002.png').convert('RGB')
x = to_tensor(img)
m = to_tensor(mask)
i = ImageChops.add(mask,img)
i = to_tensor(i)




model = create_model('vit_tiny_patch16_224', pretrained=True, num_classes= 1)
model.blocks[-1].attn.forward = my_forward_wrapper(model.blocks[-1].attn)

y = model(x.unsqueeze(0))
attn_map = model.blocks[-1].attn.attn_map.mean(dim=1).squeeze(0).detach()
cls_weight = model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(14, 14).detach()

img_resized = x.permute(1, 2, 0) * 0.5 + 0.5
map_resized = m.permute(1, 2, 0) * 0.5 + 0.5
i_resized = i.permute(1, 2, 0) * 0.5 + 0.5
cls_resized = F.interpolate(cls_weight.view(1, 1, 14, 14), (224, 224), mode='bilinear').view(224, 224)

show_img(cls_weight)
#show_img(img_resized)
#show_img(map_resized)

#show_img2(img_resized, cls_resized, alpha=0.5)
#show_img2(img_resized, map_resized, alpha=0.4)

# cls_weight_image = transforms.ToPILImage()(cls_weight)
# cls_weight_image.save('cls_weight.png')
