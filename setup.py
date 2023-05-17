import setuptools

setuptools.setup(
    name='xmem',
    version='0.0.1',
    description='Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=['torch'],
    extras_require={})
