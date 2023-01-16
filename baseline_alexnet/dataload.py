import numpy as np
import cv2
import random
import torch
import time
from torchvision.transforms import transforms

all_categories_list = ['trombone.npz', 'ear.npz', 'harp.npz', 'star.npz', 'boomerang.npz', 'hedgehog.npz',
                       'broccoli.npz',
                       'scorpion.npz', 'snake.npz', 'cat.npz', 'leaf.npz', 'toe.npz', 'mailbox.npz', 'eraser.npz',
                       'mermaid.npz', 'line.npz', 'cup.npz', 'snail.npz', 'swing set.npz', 'pants.npz', 'ladder.npz',
                       'lobster.npz', 'hexagon.npz', 'violin.npz', 'leg.npz', 'circle.npz', 'cruise ship.npz',
                       'purse.npz',
                       'bowtie.npz', 'table.npz', 'paper clip.npz', 'snowflake.npz', 'crown.npz', 'broom.npz',
                       'van.npz',
                       'donut.npz', 'ceiling fan.npz', 'tornado.npz', 'paint can.npz', 'octopus.npz', 'beard.npz',
                       'lantern.npz', 'cookie.npz', 'sock.npz', 'dragon.npz', 'eye.npz', 'fork.npz', 'soccer ball.npz',
                       'candle.npz', 'teddy-bear.npz', 'cactus.npz', 'ambulance.npz', 'car.npz', 'axe.npz',
                       'church.npz',
                       'bear.npz', 'sink.npz', 't-shirt.npz', 'skyscraper.npz', 'belt.npz', 'skateboard.npz',
                       'cello.npz',
                       'fish.npz', 'coffee cup.npz', 'finger.npz', 'waterslide.npz', 'rain.npz', 'truck.npz',
                       'ocean.npz',
                       'streetlight.npz', 'sandwich.npz', 'crayon.npz', 'birthday cake.npz', 'raccoon.npz',
                       'bridge.npz',
                       'yoga.npz', 'pickup truck.npz', 'camouflage.npz', 'bush.npz', 'parachute.npz', 'strawberry.npz',
                       'knife.npz', 'lighter.npz', 'jacket.npz', 'jail.npz', 'necklace.npz', 'hurricane.npz',
                       'rake.npz',
                       'carrot.npz', 'pizza.npz', 'oven.npz', 'computer.npz', 'toaster.npz', 'cooler.npz',
                       'motorbike.npz',
                       'pond.npz', 'potato.npz', 'sun.npz', 'dishwasher.npz', 'dumbbell.npz', 'stitches.npz',
                       'paintbrush.npz', 'wine glass.npz', 'The Eiffel Tower.npz', 'garden hose.npz', 'bus.npz',
                       'toothpaste.npz', 'baseball.npz', 'clock.npz', 'castle.npz', 'golf club.npz', 'bat.npz',
                       'tooth.npz',
                       'spreadsheet.npz', 'hockey puck.npz', 'cloud.npz', 'nail.npz', 'moon.npz', 'palm tree.npz',
                       'calendar.npz', 'blueberry.npz', 'pineapple.npz', 'mountain.npz', 'pliers.npz', 'stove.npz',
                       'basketball.npz', 'sword.npz', 'hamburger.npz', 'floor lamp.npz', 'suitcase.npz', 'dolphin.npz',
                       'watermelon.npz', 'string bean.npz', 'bathtub.npz', 'spider.npz', 'microwave.npz',
                       'hourglass.npz',
                       'baseball bat.npz', 'The Great Wall of China.npz', 'traffic light.npz', 'binoculars.npz',
                       'crab.npz',
                       'anvil.npz', 'butterfly.npz', 'power outlet.npz', 'squirrel.npz', 'drill.npz', 'chair.npz',
                       'fireplace.npz', 'foot.npz', 'squiggle.npz', 'airplane.npz', 'eyeglasses.npz', 'calculator.npz',
                       'parrot.npz', 'megaphone.npz', 'bandage.npz', 'pig.npz', 'frog.npz', 'chandelier.npz',
                       'lollipop.npz', 'vase.npz', 'lightning.npz', 'frying pan.npz', 'asparagus.npz', 'blackberry.npz',
                       'pear.npz', 'submarine.npz', 'peas.npz', 'door.npz', 'rabbit.npz', 'tree.npz', 'canoe.npz',
                       'alarm clock.npz', 'apple.npz', 'mouse.npz', 'beach.npz', 'pillow.npz', 'stop sign.npz',
                       'grass.npz',
                       'passport.npz', 'feather.npz', 'triangle.npz', 'rollerskates.npz', 'elbow.npz', 'house.npz',
                       'marker.npz', 'bird.npz', 'flip flops.npz', 'light bulb.npz', 'hot air balloon.npz', 'hat.npz',
                       'school bus.npz', 'scissors.npz', 'panda.npz', 'camera.npz', 'compass.npz', 'bicycle.npz',
                       'onion.npz', 'The Mona Lisa.npz', 'lipstick.npz', 'fan.npz', 'snowman.npz', 'swan.npz',
                       'dog.npz',
                       'nose.npz', 'umbrella.npz', 'flashlight.npz', 'saw.npz', 'square.npz',
                       'giraffe.npz',
                       'screwdriver.npz', 'remote control.npz', 'dresser.npz', 'barn.npz', 'wristwatch.npz',
                       'stairs.npz',
                       'tiger.npz', 'key.npz', 'guitar.npz', 'lighthouse.npz', 'angel.npz', 'backpack.npz',
                       'toilet.npz',
                       'roller coaster.npz', 'moustache.npz', 'helicopter.npz', 'couch.npz', 'bucket.npz',
                       'toothbrush.npz',
                       'smiley face.npz', 'arm.npz', 'firetruck.npz', 'sleeping bag.npz', 'mosquito.npz', 'steak.npz',
                       'rainbow.npz', 'shark.npz', 'kangaroo.npz', 'hot dog.npz', 'windmill.npz', 'see saw.npz',
                       'microphone.npz', 'shoe.npz', 'hand.npz', 'wheel.npz', 'camel.npz', 'cell phone.npz',
                       'monkey.npz',
                       'trumpet.npz', 'pencil.npz', 'pool.npz', 'hockey stick.npz', 'shovel.npz', 'bracelet.npz',
                       'washing machine.npz', 'banana.npz', 'wine bottle.npz', 'flamingo.npz', 'brain.npz',
                       'syringe.npz',
                       'diving board.npz', 'hammer.npz', 'zebra.npz', 'face.npz', 'television.npz', 'teapot.npz',
                       'bottlecap.npz', 'keyboard.npz', 'aircraft carrier.npz', 'matches.npz', 'knee.npz',
                       'snorkel.npz',
                       'postcard.npz', 'cake.npz', 'flower.npz', 'underwear.npz', 'cow.npz', 'elephant.npz',
                       'bench.npz',
                       'sea turtle.npz', 'book.npz', 'sailboat.npz', 'helmet.npz', 'campfire.npz', 'cannon.npz',
                       'stereo.npz', 'shorts.npz', 'duck.npz', 'police car.npz', 'envelope.npz', 'popsicle.npz',
                       'laptop.npz', 'river.npz', 'map.npz', 'telephone.npz', 'house plant.npz', 'tent.npz',
                       'picture frame.npz', 'penguin.npz', 'flying saucer.npz', 'bread.npz', 'saxophone.npz', 'bed.npz',
                       'piano.npz', 'owl.npz', 'tennis racquet.npz', 'headphones.npz', 'rifle.npz', 'skull.npz',
                       'train.npz', 'lion.npz', 'goatee.npz', 'mug.npz', 'diamond.npz', 'ice cream.npz', 'mouth.npz',
                       'octagon.npz', 'radio.npz', 'sweater.npz', 'drums.npz', 'peanut.npz', 'bee.npz', 'tractor.npz',
                       'rhinoceros.npz', 'fire hydrant.npz', 'animal migration.npz', 'spoon.npz', 'speedboat.npz',
                       'horse.npz', 'crocodile.npz', 'zigzag.npz', 'clarinet.npz', 'hospital.npz', 'bulldozer.npz',
                       'mushroom.npz', 'garden.npz', 'basket.npz', 'stethoscope.npz', 'whale.npz', 'ant.npz',
                       'grapes.npz',
                       'hot tub.npz', 'fence.npz', "sheep.npz"]

our_category_list = ["moon.npz", "airplane.npz", "fish.npz", "umbrella.npz", "train.npz",
          "spider.npz", "shoe.npz", "apple.npz", "lion.npz", "bus.npz"],

class DataSet:
    def __init__(self, root_path, mod):
        self.sketch_data_list = []
        self.sketch_npy = None
        self.sketch_data_index_list = []  
        self.sketch_number = 0  
        self.class_number = 0

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.loader(root_path, mod)

    def loader(self, root_path, mod):
        self.sketch_data_list = []
        for each in our_category_list:
            assert each in all_categories_list
        for each in all_categories_list:
            _npz = np.load(f"{root_path}/{each}", allow_pickle=True, encoding="latin1")
            self.sketch_data_list.append(_npz[mod])
            self.class_number += 1
            print(f"load {each} success, length: {len(self.sketch_data_list[-1])}")
        self.sketch_npy = np.concatenate(self.sketch_data_list, 0)
        self.sketch_data_index_list = list(range(self.sketch_npy.shape[0]))
        self.sketch_number = len(self.sketch_data_list[-1])

    def get_batch(self, batch_size):
        _random_index_list = random.choices(self.sketch_data_index_list, k=batch_size)
        X = torch.Tensor(batch_size, 3, 256, 256)
        Y = torch.Tensor(batch_size, 1)
        for index, each in enumerate(_random_index_list):
            X[index] = self.preprocess(self.draw_three(self.sketch_npy[each],
                                                       random_color=False, show=False, img_size=256))
            Y[index] = each // self.sketch_number
        return X.float(), Y.long().squeeze()

    def canvas_size_google(self, sketch):
        vertical_sum = np.cumsum(sketch[1:], axis=0) 
        xmin, ymin, _ = np.min(vertical_sum, axis=0)
        xmax, ymax, _ = np.max(vertical_sum, axis=0)
        w = xmax - xmin
        h = ymax - ymin
        start_x = -xmin - sketch[0][0] 
        start_y = -ymin - sketch[0][1]
        return [int(start_x), int(start_y), int(h), int(w)]

    def draw_three(self, sketch, window_name="google", padding=30,
                   random_color=False, time=1, show=True, img_size=256):
        thickness = int(img_size * 0.025)

        sketch = self.scale_sketch(sketch, (img_size, img_size))  
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
            if int(state) == 1:  
                first_zero = True
                if random_color:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    color = (0, 0, 0)
            pen_now += delta_x_y
        return cv2.resize(canvas, (img_size, img_size))

    def scale_sketch(self, sketch, size=(448, 448)):
        [_, _, h, w] = self.canvas_size_google(sketch)
        if h >= w:
            sketch_normalize = sketch / np.array([[h, h, 1]], dtype=np.float)
        else:
            sketch_normalize = sketch / np.array([[w, w, 1]], dtype=np.float)
        sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=np.float)
        return sketch_rescale.astype("int16")


def canvas_size_google(sketch):
    vertical_sum = np.cumsum(sketch[1:], axis=0)  
    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)
    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]  
    start_y = -ymin - sketch[0][1]
    return [int(start_x), int(start_y), int(h), int(w)]


def draw_three(sketch, window_name="google", padding=30,
               random_color=False, time=1, show=False, img_size=256):
    thickness = int(img_size * 0.025)

    sketch = scale_sketch(sketch, (img_size, img_size))  
    [start_x, start_y, h, w] = canvas_size_google(sketch=sketch)
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
        if int(state) == 1:  
            first_zero = True
            if random_color:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = (0, 0, 0)
        pen_now += delta_x_y
    return cv2.resize(canvas, (img_size, img_size))


def scale_sketch(sketch, size=(448, 448)):
    [_, _, h, w] = canvas_size_google(sketch)
    if h >= w:
        sketch_normalize = sketch / np.array([[h, h, 1]], dtype=np.float)
    else:
        sketch_normalize = sketch / np.array([[w, w, 1]], dtype=np.float)
    sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=np.float)
    return sketch_rescale.astype("int16")


if __name__ == '__main__':
    import glob
    import os
    import cv2

    for each_cate in our_category_list:
        cat_name = each_cate.replace('.npz', '')
        if not os.path.exists(f"./our_cls_test_images/{each_cate.replace('.npz', '')}"):
            os.mkdir(f"./our_cls_test_images/{cat_name}")
        npz_load = np.load(f"./datasets/{each_cate}", allow_pickle=True, encoding="latin1")["test"]
        for index, each_sketch in enumerate(npz_load):
            sketch_cv = draw_three(each_sketch, img_size=256)
            cv2.imwrite(f"./our_cls_test_images/{cat_name}/{index}.jpg", sketch_cv)
            print(f"{cat_name}, {index}")
