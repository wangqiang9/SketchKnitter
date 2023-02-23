from setuptools import setup

setup(
    name="sketch-diffusion",
    py_modules=["sketch_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
