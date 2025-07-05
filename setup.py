from setuptools import setup

setup(
    name="interactive-unet",
    version="0.1.1",
    description="An interactive segmentation tool for 3D volumetric data, with changes made by HW",
    license="BSD 2-clause",
    packages=["interactive_unet"],
    entry_points={
        "console_scripts": ["interactive-unet=interactive_unet:app"],
    },
    install_requires=[
        "torch",
        "torchvision",
        "nicegui",
        "scikit-image",
        "opencv-python",
        "lightning",
        "segmentation-models-pytorch",
        "scipy",
        "monai",
        "plotly",
        "pandas",
    ],
)
