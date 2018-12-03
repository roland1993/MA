# MA

## Image Registration Experiments
First simple experiments for non-parametric image registration, written in **MATLAB**. Inspired by [**FAIR.m**](https://github.com/C4IR/FAIR.m), but not using it. Provides:

+ ### Distance Measures
  + SSD

+ ### Regularizers
  + Diffusive Energy
  + Curvature Energy

+ ### Optimization schemes
  + Gradient descent
  + Gauß-Newton optimization
  + Armijo line search
  + Support for Multi-Level strategy

+ ### Miscellaneous
  + Derivative test (1st + 2nd order) for multivariate functions

## Primal Dual Optimization
Convex optimization experiments with first-order primal-dual algorithm by [**Chambolle & Pock**](https://hal.archives-ouvertes.fr/hal-00490826/document), written in **MATLAB**. Provides:

+ ### TV-L1 Image Denoising
+ ### TV-L2 Image Registration
+ ### TV-L1 Image Registration #####(see [A Duality Based Algorithm for TV-L1-Optical-Flow Image Registration](https://link.springer.com/chapter/10.1007/978-3-540-75759-7_62))

###### Note: Image Registration procedures use an iterative linear approximation of the template image T to achieve a convex data term.
