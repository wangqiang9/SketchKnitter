import argparse
import os
import torch as th

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
from draw_sketch import DrawSketch

def bin_pen(x, pen_break=0.005):
    result = x
    for i in range(x.size()[0]):
        for j in range(x.size()[1]):
                pen = x[i][j][2]
                if pen >= pen_break:
                    result[i][j][2] = 1
                else:
                    result[i][j][2] = 0
    return result

def main():
    args = create_argparser().parse_args()

    if os.path.exists(args.log_dir+'/test') is False:
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
            (args.batch_size, 96, 2),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample_all = th.cat((sample, pen_state), 2).cpu()
        sample_all = bin_pen(sample_all, args.pen_break)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=5000,
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
    draw = DrawSketch()
    main()
