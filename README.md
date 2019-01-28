# MA

## Image Registration Experiments
First simple experiments for non-parametric image registration, written in **MATLAB**. Provides:

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
Convex optimization experiments with first-order primal-dual algorithm by [**Chambolle & Pock**](https://hal.archives-ouvertes.fr/hal-00490826/document). Provides:

+ ### TV-L1 Image Denoising
+ ### TV-L2 and TV-L1 Image Registration
 
###### Note: Image Registration procedures use an iterative linear approximation of the image model to achieve a convex data term. Details can be found in [A Duality Based Algorithm for TV-L1-Optical-Flow Image Registration](https://link.springer.com/chapter/10.1007/978-3-540-75759-7_62).

## Nuclear Norm Experiments
+ A new distance measure for **parallel image registration** of an arbitrary number of template images to one shared reference. The rough idea is to **constrain the rank of the matrix of column-major images** (i.e. templates and reference), thus enforcing similarity between the images. Based on (and modified from) [Shape from Light Field Meets Robust PCA](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_48). 

+ **Optimization** is performed in a similiar fashion as the TV-L1 and TV-L2 registration from above, i.e. using **convex image model approximations** and applying the **primal-dual algorithm** by **Chambolle & Pock**.
