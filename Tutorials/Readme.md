# SeaPy Tutorials:
The following list of tutorials are prepared to demonstrate some of the key functionalities of the SeaPy package. The main goal is to familiarize the user with the syntax, notation and the conventions that have been used in developing the SeaPy. We have included some of the derivations that we think are handy for everyday use or might help in checking the answers. Step by step calculations are provided along with the results of the code so that the user can follow the steps and troubleshoot the possible inconsistencies. Please refer to the SeaPy documentation for a detailed list of functions, variables and definitions.

## Educational Example: Dislocations in two dimensions

![Dislocation structure](/Tutorials/511burg2.gif)

For demonstrating SeaPy's functionalities, we will utilize a very simple and educational example, that of dislocation pairs nucleating or annihilating in a thin film. It is a characteristic example, given that it can be clearly seen what the produced "fingerprints" really are, and why they represent a wealth of more information than typical imaging approaches in micromechanics.

Edge dislocations in two dimensions have an exactly solvable strain field. If one considers only the first strain invariant, for every dislocation, one considers the dislocation pressure:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;I_1(\epsilon)=\frac{\cdot(1+\nu)b}{3\pi(1-\nu)}\frac{y}{x^2+y^2}" title="First strain invariant / pressure" />
