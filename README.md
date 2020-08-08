# Neural Artistic Style Transfer 

This is a pytorch implementation of Neural Artistic Style Transfer, which contains both gradient-based methods (e.g., Gatys et al.) and feed-forward methods (e.g., Johnson et al.).

# Setup
All code is implemented in Ubuntu 16.04 with following packages:
1. Pytorch >= 1.0
2. Python >= 3.6
3. Cuda >= 9.0

Install pytorch-colors to preserve original colors of content image onto results. Please follow the README script to install it.


# Examples

## Examples from Gradient-based methods
1. results produced by a normal gradient-based neural style transfer 
<div align='center'>
  <img src='optimization/data/44.png' height='155px'>
  <img src='optimization/data/starry_night.jpg' height='155px'>
  <img src='optimization/output/44-2-starry_night.png' height='155px'>
  <br>
  <img src='optimization/data/girl_face.png' height='155px'>
  <img src='optimization/data/candy.jpg' height='155px'>
  <img src='optimization/output/girl_face-2-candy.png' height='155px'>
</div>




## Examples from Feed-forward methods

# Parameters
