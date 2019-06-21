# SeaPy: An open-source python package for the mechanical investigation of solids with unknown microstructures, using spatially resolved sequences of strain images but no constitutive equations of properties.
## https://stefanospapanikolaou.faculty.wvu.edu/research
SeaPy is a python package for performing three main tasks that are labeled as "Stability of Elasticity Analysis"(SEA):
1. Use a sequence of spatially resolved strain images to construct an elastic "fingerprint" of incipient elastic instabilities.
2. Populate the library libΨ with "fingerprint" and mechanical properties of "training" data.
3. Use image recognition to compare fingerprints and identify corresponding probabilities for reconstructing unknown microstructures.
The package includes all the necessary tools for performing SEA on a set of images. Tutorials are provided with model microstructures, libraries and plausible predictions.

![Image of method](/sea.png)

## Functions:

## Classes:


## How to Use This Package:

1. To install the development version of SeaPy, just Clone this repository:

` $ git clone https://github.com/PapStatMechMat/SeaPy.git `

2. and run the setup script.

` $ python setup.py install `

3. Import the package:

   ` >>> import SeaPy `

4. Call the function by using:

  `  >>> SeaPy.<name_of_the_function> `

5. For example to find the third elastic instability mode of a set of strain images

  `  >>> SeaPy.eim(images,3) `

6. You can also use the tools provided in this package individually by importing the functions separately. For example you may use :

` from SeaPy import <name_of_the_function> as <a_name>. `

7. Please consult the [documentation](https://drive.google.com/open?id=1Q3TxNL26vIZTEqmbR3OiJyF-KxYJDAdO) for further details.

## Prerequisites:

1. Install [Anaconda3](https://www.anaconda.com/)

## Cite SeaPy:
### S. Papanikolaou, Data-Rich, Equation-Free Predictions of Plasticity and Damage in Solids, (under review in Phys. Rev. Materials) arXiv:1905.11289 (2019)

### SeaPy on [Github](https://github.com/PapStatMechMat/SeaPy)

# Credits:
* SeaPy is written by:

[S. Papanikolaou](https://stefanospapanikolaou.faculty.wvu.edu/publications)

#### Copyright (c) 2019, Stefanos Papanikolaou.
