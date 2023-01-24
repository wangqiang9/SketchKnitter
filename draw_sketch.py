import cv2
import os
from PIL import Image
import matplotlib
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
 
 
class DrawSketch(object):
    def __init__(self):
        pass
 
    def scale_sketch(self, sketch, size=(448, 448)):
        [_, _, h, w] = self.canvas_size_google(sketch)
        if h >= w:
            sketch_normalize = sketch / np.array([[h, h, 1]], dtype=np.float)
        else:
            sketch_normalize = sketch / np.array([[w, w, 1]], dtype=np.float)
        sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=np.float)
        return sketch_rescale.astype("int16")
 
    def canvas_size_google(self, sketch):
        vertical_sum = np.cumsum(sketch[1:], axis=0)
        xmin, ymin, _ = np.min(vertical_sum, axis=0)
        xmax, ymax, _ = np.max(vertical_sum, axis=0)
        w = xmax - xmin
        h = ymax - ymin
        start_x = -xmin - sketch[0][0]
        start_y = -ymin - sketch[0][1]
        return [int(start_x), int(start_y), int(h), int(w)]
 
    def draw_three(self, sketch, random_color=False, show=False, img_size=512):
        thickness = int(img_size * 0.025)
 
        sketch = self.scale_sketch(sketch, (img_size, img_size))  # scale the sketch.
        [start_x, start_y, h, w] = self.canvas_size_google(sketch=sketch)
        start_x += thickness + 1
        start_y += thickness + 1
        canvas = np.ones((max(h, w) + 3 * (thickness + 1), max(h, w) + 3 * (thickness + 1), 3), dtype='uint8') * 255
        if random_color:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            color = (0, 0, 0)
        pen_now = np.array([start_x, start_y])
        first_zero = False
        for stroke in sketch:
            delta_x_y = stroke[0:0 + 2]
            state = stroke[2:]
            if first_zero:
                pen_now += delta_x_y
                first_zero = False
                continue
            cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
            if int(state) == 1:  # next stroke
                first_zero = True
                if random_color:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    color = (0, 0, 0)
            pen_now += delta_x_y
        if show:
            key = cv2.waitKeyEx()
            if key == 27:  # esc
                cv2.destroyAllWindows()
                exit(0)
        return cv2.resize(canvas, (img_size, img_size))
 
 
class SketchData(object):
    def __init__(self, dataPath, model="train"):
        self.dataPath = dataPath
        self.model = model

    def load(self):
        dataset_origin_list = []
        category_list = self.getCategory()
        for each_name in category_list:
            npz_tmp = np.load(f"./{self.dataPath}/{each_name}", encoding="latin1", allow_pickle=True)[self.model]
            print(f"dataset: {each_name} added.")
            dataset_origin_list.append(npz_tmp)
        return dataset_origin_list

    def getCategory(self):
        category_list = os.listdir(self.dataPath)
        return category_list
 
 
if __name__ == '__main__':
    sketchdata = SketchData(dataPath='./datasets_npz')
    category_list = sketchdata.getCategory()
    dataset_origin_list = sketchdata.load()

    for category_index in range(len(category_list)):
        sample_category_name = category_list[category_index]
        print(sample_category_name)
        save_name = sample_category_name.replace(".npz", "")
        folder = os.path.exists(f"./save_sketch/{save_name}/")
        if not folder:
            os.makedirs(f"./save_sketch/{save_name}/")
            print(f"./save_sketch/{save_name}/ is new mkdir!")
        drawsketch = DrawSketch()

        for image_index in range(10):
            # sample_sketch = dataset_origin_list[sample_category_name.index(sample_category_name)][index]
            sample_sketch = dataset_origin_list[category_list.index(sample_category_name)][image_index]
            sketch_cv = drawsketch.draw_three(sample_sketch, True)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.imshow(sketch_cv)
            plt.savefig(f"./save_sketch/{save_name}/{image_index}.jpg")
            print(f"{save_name}/{image_index}.jpg is saved!")