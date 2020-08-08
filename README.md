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
1. Results produced by a normal gradient-based neural style transfer 
<div align='center'>
  <img src='optimization/output/result_opt1.png' height='600px'>
  <img src='optimization/output/result_opt2.png' height='600px'>
</div>
2. Results produced by gradient-based method with multi-labels in corresponding masks.
<div align='center'>
  <img src='optimization/output/result_opt3.png' height='600px'>
</div>
3. Gradual transferred results
<div align='center'>
  <img src='optimization/output/result_opt4.png' height='600px'>  
</div>

## Examples from Feed-forward methods

# Parameters
