from setuptools import setup

setup(
    name='satellite-cloud-generator',
    version='0.3',    
    description='A PyTorch-based tool for simulating clouds in satellite images.',
    url='https://github.com/strath-ai/SatelliteCloudGenerator',
    author='Mikolaj Czerkawski, Christos Tachtatzis',
    author_email="mikolaj.czerkawski@esa.int",
    license='Apache 2.0',
    package_dir={"satellite_cloud_generator":"src"},
    install_requires=[
        "torch>=1.10.0",
        "torchvision",
        "kornia",
        "numpy",
        "imageio",
    ],
)
