import sys
import os
from setuptools import setup

main_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(main_dir, "SeaPy"))
import SeaPy

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='SeaPy',
    version='0.1.1',
    author='S. Papanikolaou',
    author_email='stefanos.papanikolaou@mail.wvu.edu',
    description="SeaPy is an open-source python package for investigating solids using consecutive strain images",
    long_description=read('README.md'),
    url='https://github.com/PapStatMechMat/SeaPy',
    download_url = 'https://github.com/PapStatMechMat/SeaPy/tarball/0.1.1',
    platforms='any',
    requires=['Anaconda3'],
    keywords = ['Elasticity', 'Avalanches', 'Deep Neural Networks', 'Convolutions', 'Plasticity', 'Damage'],
    classifiers=['Development Status :: 2 - Pre-Alpha', 'Topic :: Utilities'],
    license='License :: GNU-GPL',
)
