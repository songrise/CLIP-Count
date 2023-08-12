import torch
import clip
from PIL import Image
import json as js
import pickle


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
img_path = "/root/autodl-tmp/CounTR/data/images_384_VarV2/{}"
img_class_path = "/root/autodl-tmp/CounTR/data/FSC147/ImageClasses_FSC147.txt"
corpus_path = "/root/autodl-tmp/CLIPCount/util/corpus.txt"
output_path = "/root/autodl-tmp/CLIPCount/util/CLIP_caption.txt"
text_threshold = 0.6
with open(img_class_path, "r") as f:
    # 2.jpg sea shells -> [2.jpg], [sea shells]
    img_names, img_classes = zip(*[line.strip().split("\t") for line in f.readlines()])

with open(corpus_path, "r") as f:
    all_prompts = [line.strip() for line in f.readlines()]
    batch_size = 64
    all_prompt_embeddings = {}
    for i in range(0, len(all_prompts), batch_size):
        with torch.no_grad():
            text_embed = model.encode_text(clip.tokenize(all_prompts[i:i+batch_size]).to(device))
            all_prompt_embeddings.update({prompt: embed for prompt, embed in zip(all_prompts[i:i+batch_size], text_embed.cpu().numpy())})

test_cnt = 10000
caption_dict = {}
with open("util/CLIP_caption.json", "w") as f:
    for img_name, img_class in zip(img_names, img_classes):
        image = preprocess(Image.open(img_path.format(img_name))).unsqueeze(0).to(device)
        gt_text = clip.tokenize([img_class]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image).to(device)
            gt_text_features = model.encode_text(gt_text).to(device)
            
            # clip_sim = torch.cosine_similarity(image_features, gt_text_features, dim=-1)
            # text_sim = torch.cosine_similarity(gt_text_features, other_text_features, dim=-1)
            # select top-3 cosine similarity

            #first, filter out the prompts that is far from the gt_text
            close_prompt = [prompt for prompt, embed in all_prompt_embeddings.items() if torch.cosine_similarity(gt_text_features, torch.tensor(embed).to(device), dim=-1) > text_threshold]

            #then, select top-k prompt that is close to the image
            img_all_prompt_sim = torch.tensor([torch.cosine_similarity(image_features, torch.tensor(all_prompt_embeddings[prompt]).to(device), dim=-1) for prompt in close_prompt])
            top_k_idx = torch.argsort(img_all_prompt_sim, descending=True)[:5]
            top_k_prompt = [list(all_prompt_embeddings.keys())[idx] for idx in top_k_idx]


            
            # output to json
            f.write(js.dumps({"img_name": img_name, "img_class": img_class, "top_k_prompt": top_k_prompt}) + "\n")
            caption_dict[img_name] = top_k_prompt
            test_cnt -= 1
            if test_cnt%100 == 0:
                print(test_cnt)
            if test_cnt == 0:
                break

pickle.dump(caption_dict, open("util/CLIP_caption.pkl", "wb"))