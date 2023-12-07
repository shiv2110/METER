# import gradio as gr
import torch
import cv2
import copy
import time
import requests
import io
import numpy as np
import re
import json
import urllib.request
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pymatting.util.util import row_sum

from PIL import Image

from meter.config import ex
from meter.modules import METERTransformerSS

from meter.transforms import pixelbert_transform, vit_transform, clip_transform
from meter.datamodules.datamodule_base import get_pretrained_tokenizer


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    loss_names = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 1,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
    }
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    with urllib.request.urlopen(
        "https://github.com/dandelin/ViLT/releases/download/200k/vqa_dict.json"
    ) as url:
        id2ans = json.loads(url.read().decode())

    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    model = METERTransformerSS(_config)
    model.setup("test")
    model.eval()

    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)

    def infer(url, text):
        try:
            if "http" in url:
                res = requests.get(url)
                image = Image.open(io.BytesIO(res.content)).convert("RGB")
            else:
                image = Image.open(url)
            orig_shape = np.array(image).shape
            # img = pixelbert_transform(size=384)(image)
            # img = vit_transform(size=224)(image)
            img = clip_transform(size=288)(image)
            print("vit transformed image shape: {}".format(img.shape))
            img = img.unsqueeze(0).to(device)

        except:
            return False

        batch = {"text": [text], "image": [img]}

        with torch.no_grad():
            encoded = tokenizer(batch["text"])
            batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
            ret = model.infer(batch)
            vqa_logits = model.vqa_classifier(ret["cls_feats"])

        answer = id2ans[str(vqa_logits.argmax().item())]
        return [np.array(image), answer], ret['image_feats'][0], img
    


    # inputs = [
    #     gr.inputs.Textbox(
    #         label="Url of an image.",
    #         lines=5,
    #     ),
    #     gr.inputs.Textbox(label="Question", lines=5),
    # ]
    # outputs = [
    #     gr.outputs.Image(label="Image"),
    #     gr.outputs.Textbox(label="Answer"),
    # ]

    # interface = gr.Interface(
    #     fn=infer,
    #     inputs=inputs,
    #     outputs=outputs,
    #     server_name="0.0.0.0",
    #     server_port=8888,
    #     examples=[
    #         [
    #             "https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg",
    #             "What is the color of the flower?",
    #         ],
    #         [
    #             "https://computing.ece.vt.edu/~harsh/visualAttention/ProjectWebpage/Figures/vqa_1.png",
    #             "What is the mustache made of?",
    #         ],
    #         [
    #             "https://computing.ece.vt.edu/~harsh/visualAttention/ProjectWebpage/Figures/vqa_2.png",
    #             "How many slices of pizza are there?",
    #         ],
    #         [
    #             "https://computing.ece.vt.edu/~harsh/visualAttention/ProjectWebpage/Figures/vqa_3.png",
    #             "Does it appear to be rainy?",
    #         ],
    #     ],
    # )

    # interface.launch(debug=True)
    result, feats, image = infer('../../nii_depressed.jpg', 'Did the boy wear glasses?')
    # result, feats, image = infer("https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg", "What is the color of the flower?")
    PATCH_SIZE = 16
    # feats = feats.unsqueeze(dim = 0)
    print("Answer: {}".format(result[1]))
    # print("Feature shape: {} | orig_img shape: {}".format(feats.shape, orig_shape))
    feats = F.normalize(feats, p = 2, dim = -1)

    # H_patch, W_patch = orig_shape[0]//PATCH_SIZE, orig_shape[1]//PATCH_SIZE
    H_patch, W_patch = 18, 18

    H_pad_lr, W_pad_lr = H_patch, W_patch
    # feats = F.interpolate(
    #         feats.T.reshape(1, -1,  H_patch, W_patch), 
    #         size=(H_pad_lr, W_pad_lr), mode='bilinear', align_corners=False
    # ).reshape(-1, H_pad_lr * W_pad_lr).T[1:]
    W_feat = (feats @ feats.T)

    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max() 
    W_feat = W_feat.cpu().numpy()

    def get_diagonal (W):
        D = row_sum(W)
        D[D < 1e-12] = 1.0  # Prevent division by zero.
        D = diags(D)
        return D
    
    D = np.array(get_diagonal(W_feat).todense())

    L = D - W_feat
    eigenvalues, eigenvectors = eigsh(L, k = 5, which = 'LM', sigma = 0)
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
    image_relevance = eigenvectors[1, 1:] 


    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=288, mode='bilinear')
    image_relevance = image_relevance.reshape(288, 288).numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())


    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam


    image = image[0].permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)


    # plt.imshow(image_relevance)
    plt.imshow(vis)
    plt.show()
    # fiedel_ev = eigenvectors[1].numpy().reshape(H_patch, W_patch)
    # plt.imshow(fiedel_ev)
    # plt.show()
    
