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
from spectral.get_fev import get_grad_eigs, get_eigs, avg_heads


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

    model = METERTransformerSS(_config)
    model.setup("test")
    model.eval()
    # model.zero_grad()

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

        # with torch.no_grad():
        encoded = tokenizer(batch["text"])
        # print(batch['text'])
        text_tokens = tokenizer.tokenize(batch["text"][0])
        print(text_tokens)
        batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
        ret = model.infer(batch)
        # ret = model.forward(batch)
        # print(type(output))
        # ret = model.infer_relevance_maps(batch)
        # print(f"Ret cls_feats:::::::::::::::::::::::::::::: {ret['cls_feats'].shape}")
        vqa_logits = model.vqa_classifier(ret["cls_feats"])
        # print(f"{vqa_logits.shape}")

        # print( model.cross_modal_text_layers[0].crossattention.self.get_attn_gradients().detach() )
        answer = id2ans[str(vqa_logits.argmax().item())]
        output = vqa_logits
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * output) #baka

        model.zero_grad()
        one_hot.backward(retain_graph=True)
        return answer, ret['all_image_feats'][0], ret['all_text_feats'][0], img, text_tokens
    


    question = "What is she holding?"
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
    # question = "What is the time on the clock?"
    # question = "What animal is the candle?"
    # question = "What traffic sign is it?"
    # question = "is the text '02' in the image?"
    # question = "Where is the girl sitting?"
    # question = "What is on the right of the road?"
    # question = "How many street lights are there?"
    # question = "What make is the laptop?"





    # result, all_image_feats, all_text_feats, image, text_tokens = infer('../../nii_depressed.jpg', question)
    result, all_image_feats, all_text_feats, image, text_tokens = infer('images/skii.jpg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/clock_owl.jpg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/sign.jpg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/stop_sign.jpeg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/0_0_QU04029757_hr.png', question)



    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/swim.jpg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/cows.jpg', question) #localisation
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/weird_dj.jpg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/nee-sama.jpeg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/bird.jpg', question) #localisation
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/train.jpg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer("D:/Thesis_2023-24/data/root/val2014/COCO_val2014_000000395344.jpg", question)

    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/shiv.png', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/demon.png', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/time.jpg', question)

    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/buildings.jpg', question)
    # result, all_image_feats, all_text_feats, image, text_tokens = infer('images/bedroom.jpg', question)






    # print(f"Text feats shape: {text_feats.shape}")
    # print(f"image feats shape: {image_feats.shape}")



    print(f"QUESTION: {question}")
    print("Answer: {}".format(result))
    # feats = feats[1:, :]


    def get_eigen (feat_list, modality, how_many = None):
        fevs = []
        for i, feats in enumerate(feat_list):
        #     if modality == 'image':
        #         # print(f"FEATS SHAPE: {feats.size()}")
        #         feats = F.normalize(feats.detach(), p = 2, dim = -1)[1:, :]
        #         # feats = feats[0][1:, :]
        #     else:
        #         # feats = feats[0][1:-1]
        #         feats = F.normalize(feats.detach(), p = 2, dim = -1)[1:-1]

        #     W_feat = (feats @ feats.T)

        #     W_feat = (W_feat * (W_feat > 0))
        #     W_feat = W_feat / W_feat.max() 
        #     W_feat = W_feat.cpu().detach().numpy()

        #     def get_diagonal (W):
        #         D = row_sum(W)
        #         D[D < 1e-12] = 1.0  # Prevent division by zero.
        #         D = diags(D)
        #         return D
            
        #     D = np.array(get_diagonal(W_feat).todense())

        #     L = D - W_feat
        # # print(L)
        # # print("here")
        # # L[ np.isnan(L) ] = 0
        # # L[ L == np.inf ] = 0
        #     try:
        #         eigenvalues, eigenvectors = eigs(L, k = 5, which = 'LM', sigma = -0.5, M = D)
        #     except:
        #         try:
        #             eigenvalues, eigenvectors = eigs(L, k = 5, which = 'SM', sigma = -0.5, M = D)
        #         except:
        #             eigenvalues, eigenvectors = eigs(L, k = 5, which = 'LM', M = D)


        


        #     eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
        #     n_tuple = torch.kthvalue(eigenvalues.real, 2)
        #             # print(f"N_Tuple: {n_tuple.indices}")
        #     fev_idx = n_tuple.indices
        #     fev = torch.abs(eigenvectors[fev_idx])
            fev = get_eigs(feats, modality, how_many)
            if modality == "text":
                fev = fev[1:-1]
            if modality == 'image':
                grad = model.cross_modal_image_layers[i].attention.self.get_attn_gradients().detach()[:, :, 1:, 1:]
                cam = model.cross_modal_image_layers[i].attention.self.get_attention_map().detach()[:, :, 1:, 1:]
                cam = avg_heads(cam, grad)

                # grad = grad[:, :, 1:, 1:]
                # grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                # grad = grad.clamp(min=0).mean(dim=0)
                # # print(f"GRAD SHAPE: {grad.size()}")
                fev = fev.to(model.device)
                fev = cam @ fev.unsqueeze(1)
                fev = fev[:, 0]
            else:
                grad = model.cross_modal_text_layers[i].attention.self.get_attn_gradients().detach()[:, :, 1:-1, 1:-1]
                cam = model.cross_modal_text_layers[i].attention.self.get_attention_map().detach()[:, :, 1:-1, 1:-1]
                cam = avg_heads(cam, grad)

                # grad = grad[:, :, 1:-1, 1:-1]
                # grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                # grad = grad.clamp(min=0).mean(dim=0)
                # # print(f"GRAD SHAPE: {grad.size()}")
                fev = fev.to(model.device)
                fev = cam @ fev.unsqueeze(1)
                fev = fev[:, 0]
                # # if modality == 'text':
                #     # print(grad)
                fev = torch.cat( ( torch.zeros(1), fev, torch.zeros(1) ) )
            # fev = get_grad_eigs(feats, modality, grad, model.device, how_many)
            fevs.append( torch.abs(fev) )
        return fevs


    # print(f"ALL IMAGE FEATS: {len(all_image_feats)}")
    image_relevances = get_eigen(all_image_feats, "image", 5)
    image_relevance = torch.stack(image_relevances, dim=0).sum(dim=0)

    # print()
    text_relevances = get_eigen(all_text_feats, "text", 5)
    text_relevance = torch.stack(text_relevances, dim=0).sum(dim=0)


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
    axs[0].set_title('(Spectral + Grad + Attn) Image Relevance')

    ti = axs[1].imshow(text_relevance.unsqueeze(dim = 0).numpy())
    axs[1].set_title("(Spectral + Grad + Attn) Word Impotance")
    plt.sca(axs[1])
    plt.xticks(np.arange(len(text_tokens) + 2), [ '[CLS]' ] + text_tokens + [ '[SEP]' ])
    # plt.sca(axs[1])
    plt.colorbar(ti, orientation = "horizontal", ax = axs[1])

    # axs[1].axis('off')
    # axs[1].set_title('masked')


    # plt.imshow(vis)
    plt.show()


    # fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 5))

    # axs[0][0].imshow(image)
    # axs[0][0].axis('off')
    # axs[0][0].set_title('Original Image')


    # axs[0][1].imshow(vis)
    # axs[0][1].axis('off')
    # axs[0][1].set_title('(Spectral + Grad) Image Relevance')

    # ti = axs[1][0].imshow(text_relevance.unsqueeze(dim = 0).numpy())
    # axs[1][0].set_title("(Spectral + Grad) Word Impotance")
    # plt.sca(axs[1][0])
    # plt.xticks(np.arange(len(text_tokens) + 2), ['[CLS]'] + text_tokens + ['[SEP]'])
    # # plt.sca(axs[1])
    # plt.colorbar(ti, orientation = "horizontal", ax = axs[1][0])

    # plt.show()

    
