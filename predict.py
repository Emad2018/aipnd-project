import predict_helper as phelp
import json
import torch
from PIL import Image

in_args = phelp.get_input_args()
print(in_args)
with open(in_args.category_names, 'r') as f:
    cat_to_name = json.load(f)
imagepath = in_args.input
model_loaded = phelp.load_checkpoint(in_args.ckpdir, in_args.arch)
if in_args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (device == "cpu"):
        print("not found gpu,.... running on cpu")
else:
    device = "cpu"

probability, index = phelp.predict(
    imagepath, model_loaded, topk=in_args.topk, device=device)
pil_image = Image.open(imagepath)
folder = [model_loaded.class_to_idx[int(index[0][i])]
          for i in range(len(index[0]))]
name = [cat_to_name[str(folder[i])] for i in range(len(folder))]
pb = list(probability[0].cpu().detach().numpy())
print("1 st class name={}...... Probability={}".format(name[0], pb[0]))
print("2 nd class name={}...... Probability={}".format(name[1], pb[1]))
print("3 rd class name={}...... Probability={}".format(name[2], pb[2]))
print("4 fth class name={}..... Probability={}".format(name[3], pb[3]))
print("5 fith class name={}.... Probability={}".format(name[4], pb[4]))
