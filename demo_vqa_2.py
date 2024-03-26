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
from scipy.sparse import diags, csr_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pymatting.util.util import row_sum

from PIL import Image

from meter.config import ex
from meter.modules import METERTransformerSS

from meter.transforms import vit_transform, clip_transform, clip_transform_randaug
from meter.datamodules.datamodule_base import get_pretrained_tokenizer
from scipy.stats import skew


@ex.automain


def main(_config):


    _config = copy.deepcopy(_config)

    loss_names = {
        "itm": 0,
        "mlm": 1,
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

    IMG_SIZE = 576

    def infer(url, text):
        try:
            if "http" in url:
                res = requests.get(url)
                image = Image.open(io.BytesIO(res.content)).convert("RGB")
            else:
                image = Image.open(url)
            orig_shape = np.array(image).shape
            img = clip_transform(size=IMG_SIZE)(image)
            # img = vit_transform(size=IMG_SIZE)(image)
            # img = clip_transform_randaug(size=IMG_SIZE)(image)
            # print("transformed image shape: {}".format(img.shape))
            img = img.unsqueeze(0).to(device)

        except:
            return False

        batch = {"text": [text], "image": [img]}

        with torch.no_grad():
            encoded = tokenizer(batch["text"])
            # print(batch['text'])
            text_tokens = tokenizer.tokenize(batch["text"][0])
            print(text_tokens)
            batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
            ret = model.infer(batch)
            vqa_logits = model.vqa_classifier(ret["cls_feats"])

        answer = id2ans[str(vqa_logits.argmax().item())]

        return answer, ret['all_image_feats'], ret['all_text_feats'], img, text_tokens
    


    # question = "What is the colour of her pants?"
    # question = "Does he have earphones plugged in?"
    question = "Does he have spectacles?"
    # question = "Is there an owl?"
    # question = "Is the man swimming?"
    # question = "What animals are shown?"
    # question = "What animal hat did she wear?"

    # result, all_image_feats, all_text_feats, image, text_tokens = infer('../../nii_depressed.jpg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/skii.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/clock_owl.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/swim.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/cows.jpg', question)
    result, all_image_feats, all_text_feats, image, text_tokens = infer('images/weird_dj.jpg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/nee-sama.jpeg', question)
    # print(f"Text feats shape: {text_feats.shape}")


    print(f"QUESTION: {question}")
    print("Answer: {}".format(result))
    # feats = feats[1:, :]

    # print(all_image_feats[0], all_image_feats[1])
    def get_eigen (feats_list, modality):
        fevs = []
        for feats in feats_list:
            if modality == 'image':
                feats = feats[0][1:, :]
            else:
                feats = feats[0][1:-1]
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

            try:
                eigenvalues, eigenvectors = eigsh(L, k = 5, which = 'LM', sigma = 0, M = D)
            except:
                try:
                    eigenvalues, eigenvectors = eigsh(L, k = 5, which = 'LM', sigma = 0)
                except:
                    eigenvalues, eigenvectors = eigsh(L, k = 5, which = 'SM', M = D)
            


            eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
            fevs.append(eigenvectors[1])
            # break
        return fevs
    # # image_relevance = eigenvectors[1, 1:] 
    image_relevances = get_eigen(all_image_feats, "image")

    # print()
    text_relevances = get_eigen(all_text_feats, "text")
    # print(text_relevance)
    # text_relevance = torch.abs(text_relevance)
    # plt.title("Spectral Approach word impotance")
    # plt.xticks(np.arange(len(text_tokens)), text_tokens)
    # plt.imshow(text_relevance.unsqueeze(dim = 0).numpy())
    # plt.colorbar(orientation = "horizontal")

    # skew_vec = []
    # for obj_feat in W_feat:
    #     skew_vec.append(skew(obj_feat))
        
    # skew_vec = np.array(skew_vec)

    # fev, nfev = eigenvectors[1], (eigenvectors[1] * -1)
    # k1, k2 = fev.topk(k = 1).indices[0], nfev.topk(k = 1).indices[0]

    # image_relevance = nfev
    # if skew_vec[k1] <= 0 and skew_vec[k2] > 0:
    #     image_relevannce = fev
    # elif skew_vec[k1] > 0 and skew_vec[k2] <= 0:
    #     image_relevannce = nfev
    # elif skew_vec[k1] > skew_vec[k2]:
    #     image_relevannce = fev
    # else:
    #     image_relevannce = nfev

    for j, (image_relevance, text_relevance) in enumerate(zip(image_relevances, text_relevances)):
        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=IMG_SIZE, mode='bilinear')
        image_relevance = image_relevance.reshape(IMG_SIZE, IMG_SIZE).cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())


        def show_cam_on_image(img, mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            return cam


        image1 = image[0].permute(1, 2, 0).cpu().numpy()
        image1 = (image1 - image1.min()) / (image1.max() - image1.min())
        vis = show_cam_on_image(image1, image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)


        fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
        axs[0].imshow(vis)
        axs[0].axis('off')
        axs[0].set_title('Spectral Approach Image Relevance ' + str(j))

        ti = axs[1].imshow(text_relevance.unsqueeze(dim = 0).numpy())
        axs[1].set_title("Spectral Approach Word Impotance " + str(j))
        plt.sca(axs[1])
        plt.xticks(np.arange(len(text_tokens)), text_tokens)
        # plt.sca(axs[1])
        plt.colorbar(ti, orientation = "horizontal", ax = axs[1])

    # axs[1].axis('off')
    # axs[1].set_title('masked')


    # plt.imshow(vis)
    plt.show()

    
