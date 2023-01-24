from alexnet_eval.alexnet_quickdraw import AlexNet
from alexnet_eval.dataload import all_categories_list, draw_three
import torch
import torchvision
from PIL import Image

model = AlexNet(all_categories_list.__len__())
print(all_categories_list)

model.cuda().eval()
model_params = torch.load("")
model.load_state_dict(model_params)

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

category_list_varli = []
result_mean = { 'top1 acc': 0,
                'top3 acc': 0,
                'top5 acc': 0,
                'top10 acc': 0
              }

if __name__ == "__main__":
    # Ours
    for each_cate in category_list_varli:
        each_name = each_cate.replace(".npz", "")
        right_answer = all_categories_list.index(each_cate)
        acc_dict = {
            1:0,
            3:0,
            5:0,
            10:0,
        }
        RANGE = 200
        for each_image_index in range(RANGE):
            image_cv = Image.open(f"../save_sketch/{each_name}/{each_image_index}.png").convert('RGB')
            image_input = preprocess(image_cv)
            image_input = image_input.unsqueeze(0)
            result = model(image_input.cuda(0))
            value, predicted = torch.topk(result.softmax(1), 10)
            for k in acc_dict:
                if right_answer in predicted[0, :k]:
                    acc_dict[k] += 1
        print(f"{each_name} acc:")
        for k in acc_dict:
            print(f"top{k} acc: {acc_dict[k] / RANGE}")
            result_mean[f"top{k} acc"] += (acc_dict[k] / RANGE)
