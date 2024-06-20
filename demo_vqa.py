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
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import diags, csr_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pymatting.util.util import row_sum

from PIL import Image

from meter.config import ex
from meter.modules import METERTransformerSS

from meter.transforms import vit_transform, clip_transform, clip_transform_randaug, pixelbert_transform
from meter.datamodules.datamodule_base import get_pretrained_tokenizer
from scipy.stats import skew

# from ExplanationGenerator import GenerateOurs
from spectral.get_fev import get_eigs


# @ex.automain


def main1(_config, item, model=None, viz=True, is_pert=False, tokenizer=None):

    if is_pert:
        img_path = item['img_id'] + '.jpg'
        question = item['sent']
    else:
        img_path, question = item

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

    if not is_pert:
        tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    # with urllib.request.urlopen(
    #     "https://github.com/dandelin/ViLT/releases/download/200k/vqa_dict.json"
    # ) as url:
    #     id2ans = json.loads(url.read().decode())

    url = 'vqa_dict.json'
    f = open(url)
    id2ans = json.load(f)

    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    if not is_pert:
        model = METERTransformerSS(_config)
        model.setup("test")
        model.eval()
    # model.zero_grad()

    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    # model.to(device)

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
            # img = pixelbert_transform(size=IMG_SIZE)(image)

            # img = vit_transform(size=IMG_SIZE)(image)
            # img = clip_transform_randaug(size=IMG_SIZE)(image)
            # print("transformed image shape: {}".format(img.shape))
            img = img.unsqueeze(0).to(device)

        except:
            return False

        batch = {"text": [text], "image": [img]}

        # with torch.no_grad():
        encoded = tokenizer(batch["text"])
        # print(batch['text'])
        text_tokens = tokenizer.tokenize(batch["text"][0])
        # print(text_tokens)
        batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
        if not is_pert:
            ret = model.infer(batch)
        else:
            ret = model.infer_mega(batch)
        # print(type(output))
        # ret = model.infer_relevance_maps(batch)
        # print(f"Ret cls_feats:::::::::::::::::::::::::::::: {ret['cls_feats'].shape}")
        vqa_logits = model.vqa_classifier(ret["cls_feats"])
        # print(f"{vqa_logits.shape}")

        # print( model.cross_modal_text_layers[0].crossattention.self.get_attn_gradients().detach() )
        answer = id2ans[str(vqa_logits.argmax().item())]
        # ours = GenerateOurs(model)
        # R_t_t, 

        # print(f"{ret['image_feats'][0].shape, ret['']}")
        return answer, ret['image_feats'][0], ret['text_feats'][0], img, text_tokens
    


    # question = "What is she holding?"
    # question = "Does he have earphones plugged in?"
    # question = "Does he have spectacles?"
    # question = "Is there an owl?"
    # question = "Is the man swimming?"
    # question = "What animals are shown?"
    # question = "What animal hat did she wear?"
    # question = "What is the colour of the bird's feet?"
    # question = "is there a lamppost?"
    # question = "Is there a laptop?"
    # question = "Did she wear a wristwatch?"
    # question = "What is the girl in white doing?"
    # question = "Is there construction going on?"
    # question = "How many street lights are there?"
    # question = "What is the bird in the image?"
    # question = "Where is the girl sitting?"
    # question = "What is the time on the clock?"
    # question = "What does the sign board say?"
    # question = "What traffic sign is it?"
    # question = "is the text '02' in the image?"
    # question = "What make is the laptop?"
    # question = "Is the power connected to the laptop?"
    # question = "Is he wearing glasses?"
    # question = "What animal is the candle?"





    # result, image_feats, text_feats, image, text_tokens = infer('../../nii_depressed.jpg', question)
    result, image_feats, text_feats, image, text_tokens = infer(img_path, question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/clock_owl.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/time.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/buildings.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/swim.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/cows.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/weird_dj.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/shore.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/bedroom.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/nee-sama.jpeg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/bird.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/train.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/shiv.png', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/laptop.jpg', question)
    # result, image_feats, text_feats, image, text_tokens = infer("D:/Thesis_2023-24/data/root/val2014/COCO_val2014_000000395344.jpg", question)

    # result, image_feats, text_feats, image, text_tokens = infer('images/demon.png', question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/stop_sign.jpeg', question)
    # result, image_feats, text_feats, image, text_tokens = infer("D:/Thesis_2023-24/data/root/val2014/COCO_val2014_000000481214.jpg", question)
    # result, image_feats, text_feats, image, text_tokens = infer('images/0_0_QU04029757_hr.png', question)


    # print(f"Text feats shape: {text_feats.shape}")
    # print(f"image feats shape: {image_feats.shape}")



    # print(f"QUESTION: {question}")
    # print("Answer: {}".format(result))
    # feats = feats[1:, :]

    image_relevance = get_eigs(image_feats, "image", 5, device).to(device)
    text_relevance = get_eigs(text_feats, "text", 5, device).to(device)
    # print(text_relevance)

    if viz:
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


        image = image[0].permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        vis = show_cam_on_image(image, image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)


        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 5))

        axs[0][0].imshow(image)
        axs[0][0].axis('off')
        axs[0][0].set_title('Original Image')


        axs[0][1].imshow(vis)
        axs[0][1].axis('off')
        axs[0][1].set_title('Spectral Approach Image Relevance')

        ti = axs[1][0].imshow(text_relevance.unsqueeze(dim = 0).numpy())
        axs[1][0].set_title("Spectral Approach Word Impotance")
        plt.sca(axs[1][0])
        plt.xticks(np.arange(len(text_tokens) + 2), ['[CLS]'] + text_tokens + ['[SEP]'])
        # plt.sca(axs[1])
        plt.colorbar(ti, orientation = "horizontal", ax = axs[1][0])

        plt.show()
    
    return text_relevance, image_relevance

    
if __name__ == '__main__':
    @ex.automain
    def main (_config):
        item = ('images/shiv.png', "Is there a laptop?")
        R_t_t, R_t_i = main1(_config, item, viz = True)
        # print(conf)