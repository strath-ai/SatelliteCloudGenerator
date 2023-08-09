from setuptools import setup

setup(
    name='satellite_cloud_generator',
    version='0.3',    
    description='A PyTorch-based tool for simulating clouds in satellite images.',
    url='https://github.com/strath-ai/SatelliteCloudGenerator',
    author='Mikolaj Czerkawski, Christos Tachtatzis',
    author_email="mikolaj.czerkawski@strath.ac.uk",
    license='Apache 2.0',
    packages=['satellite_cloud_generator', 'satellite_cloud_generator.LocalGaussianBlur', 'satellite_cloud_generator.LocalGaussianBlur.src'],
    install_requires=[
        "torch>=1.10.0",
        "torchvision",
        "kornia",
        "numpy",
        "imageio",
    ],
)