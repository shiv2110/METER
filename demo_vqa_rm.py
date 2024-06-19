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

from meter.transforms import vit_transform, clip_transform, clip_transform_randaug
from meter.datamodules.datamodule_base import get_pretrained_tokenizer
from scipy.stats import skew

from ExplanationGenerator import GenerateOurs
import sys


# @ex.automain


def main1(_config, item, model=None, viz=True, is_pert=False, tokenizer=None):
    print(item)
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

    method_type = _config["method_name"]

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

        # with torch.no_grad():
        encoded = tokenizer(batch["text"])
        # print(batch['text'])
        text_tokens = tokenizer.tokenize(batch["text"][0])
        # print(text_tokens)
        batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
        # ret = model.infer(batch)
        if not is_pert:
            ret = model.infer(batch)
        else:
            ret = model.infer_mega(batch)

        vqa_logits = model.vqa_classifier(ret["cls_feats"])
        # print(f"{vqa_logits.shape}")

        output = vqa_logits
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if is_pert:
            one_hot = torch.sum(one_hot.cuda() * output) #baka
        else:
            one_hot = torch.sum(one_hot * output) 

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        # print( model.cross_modal_text_layers[0].crossattention.self.get_attn_gradients().detach() )
        answer = id2ans[str(vqa_logits.argmax().item())]
        ours = GenerateOurs(model = model, normalize_self_attention=False, apply_self_in_rule_10=True)
        if method_type == "rm":
            R_t_t, R_t_i = ours.generate_relevance_maps( ret['text_feats'][0].shape[0], ret['image_feats'][0].shape[0], device )
        # elif method_type == "transformer_attr":
            # R_t_t, R_t_i = ours.generate_transformer_attr( ret['text_feats'][0].shape[0], ret['image_feats'][0].shape[0], device )
        elif method_type == "attn_gradcam":
            R_t_t, R_t_i = ours.generate_attn_gradcam( ret['text_feats'][0].shape[0], ret['image_feats'][0].shape[0], device )
        elif method_type == "rollout":
            R_t_t, R_t_i = ours.generate_rollout( ret['text_feats'][0].shape[0], ret['image_feats'][0].shape[0], device )

        elif method_type == "raw_attn":
            R_t_t, R_t_i = ours.generate_raw_attn( ret['text_feats'][0].shape[0], ret['image_feats'][0].shape[0], device )
        # elif method_type == "lrp":
            # model.relprop(torch.tensor(one_hot_vector).to(output.device))
            # R_t_t, R_t_i = ours.generate_lrp( ret['text_feats'][0].shape[0], ret['image_feats'][0].shape[0], device )
        else:
            print(f"Methods available: attn_gradcam, rm, rollout, raw_attn")
            sys.exit()


        return answer, R_t_t, R_t_i, img, text_tokens
    


    # question = "What is she holding?"
    # question = "Does he have earphones plugged in?"
    # question = "Does he have spectacles?"
    # question = "Is there an owl?"
    # question = "What is the colour of the man's shirt?"
    # question = "What animals are shown?"
    # question = "What animal hat did she wear?"
    # question = "What is the colour of the bird's feet?"
    # question = "is there a train?"
    # question = "Is there a laptop?"
    # question = "Did she wear a wristwatch?"
    # question = "What is the girl in white doing?"
    # question = "is the text '02' in the image?"
    # question = "What make is the laptop?"
    # question = "How many street lights are there?"
    # question = "Where is the girl sitting?"
    # question = "Is the power connected to the laptop?"
    # question = "Is there construction going on?"




    # result, R_t_t, R_t_i, image, text_tokens = infer('../../nii_depressed.jpg', question)
    result, R_t_t, R_t_i, image, text_tokens = infer(img_path, question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/clock_owl.jpg', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/swim.jpg', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/cows.jpg', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/weird_dj.jpg', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/nee-sama.jpeg', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/bird.jpg', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/train.jpg', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/shiv.png', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/0_0_QU04029757_lr.png', question)

    # result, R_t_t, R_t_i, image, text_tokens = infer('images/demon.png', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('D:/Thesis_2023-24/data/root/val2014/COCO_val2014_000000395344.jpg', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/buildings.jpg', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/bedroom.jpg', question)
    # result, R_t_t, R_t_i, image, text_tokens = infer('images/laptop.jpg', question)

    # print(f"Text feats shape: {text_feats.shape}")
    # print(f"image feats shape: {image_feats.shape}")


    # print(f"ANSWER: {result}")
    # print(f"R_t_i shape: {R_t_i[0].shape}")
    # print(f"R_t_t shape: {R_t_t[0].shape}")

    image_relevance = R_t_i[0][1:].detach()
    text_relevance = R_t_t[0].detach()

    # print(image_relevance, text_relevance)
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




        fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
        axs[0].imshow(vis)
        axs[0].axis('off')
        axs[0].set_title(method_type + " image relevance")

        ti = axs[1].imshow(text_relevance.unsqueeze(dim = 0).numpy())
        axs[1].set_title(method_type + " word importance")
        plt.sca(axs[1])
        plt.xticks(np.arange(len(text_tokens) + 2), [ '[CLS]' ] + text_tokens + [ '[SEP]' ])
        # plt.sca(axs[1])
        plt.colorbar(ti, orientation = "horizontal", ax = axs[1])

        # axs[1].axis('off')
        # axs[1].set_title('masked')


        # plt.imshow(vis)
        plt.show()
    return text_relevance, image_relevance


if __name__ == '__main__':
    @ex.automain
    def main (_config):
        item = ('images/buildings.jpg', "Is there construction going on?")
        # print(item)
        R_t_t, R_t_i = main1(_config, item, viz = True)
        # print(conf)
    
