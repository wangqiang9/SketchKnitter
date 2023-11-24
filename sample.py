import random
import argparse
import os
import torch as th
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import cv2

from sketch_diffusion import dist_util, logger
from sketch_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    # different modes
    create_model_and_diffusion,
    # create_model_and_diffusion_acc
    # create_model_and_diffusion_noise,
    add_dict_to_argparser,
    args_to_dict,
)


def canvas_size_google(sketch):
    vertical_sum = np.cumsum(sketch[1:], axis=0)  

    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)

    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]  
    start_y = -ymin - sketch[0][1]
    return [int(start_x), int(start_y), int(h), int(w)]


def scale_sketch(sketch, size=(448, 448)):
    [_, _, h, w] = canvas_size_google(sketch)
    if h >= w:
        sketch_normalize = sketch / np.array([[h, h, 1]], dtype=float)
    else:
        sketch_normalize = sketch / np.array([[w, w, 1]], dtype=float)
    sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=float)
    return sketch_rescale.astype("int16")


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


def bin_pen(x, pen_break=0.005):
    result = x
    for i in range(x.size()[0]):
        for j in range(x.size()[1]):
                pen = x[i][j][2]
                if pen >= pen_break:
                    result[i][j][2] = 1
                else:
                    result[i][j][2] = 0
    return result[:, :, :3]

def main():
    args = create_argparser().parse_args()

    if not os.path.exists(args.log_dir+'/test'):
        os.makedirs(args.log_dir+'/test')
    args.log_dir = args.log_dir + '/test'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    dist_util.setup_dist()
    logger.configure(args.log_dir)

    logger.log("creating model and diffusion...")
    # different modes, if noise or acc method, please specify 'data', 'raster', and 'loss'.
    model, diffusion = create_model_and_diffusion(
    #model, diffusion = create_model_and_diffusion_acc(
    #model, diffusion = create_model_and_diffusion_noise(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, pen_state, _ = sample_fn(
            model,
            (args.batch_size, args.image_size, 2),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample_all = th.cat((sample, pen_state), 2).cpu()
        sample_all = bin_pen(sample_all, args.pen_break)

        for sample in sample_all:
            sample = sample.numpy()
            sketch_cv = draw_three(sample, img_size=256)

            # Convert the image to Torch tensor
            tensor = transforms.ToTensor()(sketch_cv)

            all_images.append(tensor)

        np.savez(os.path.join(args.save_path, 'result.npz'), sample_all)

    save_image(th.stack(all_images), os.path.join(args.save_path, 'output.png'))

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50,
        batch_size=16,
        use_ddim=False,
        model_path="",
        log_dir='',
        save_path="",
        pen_break=0.5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
