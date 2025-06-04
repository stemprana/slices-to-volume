# slices-to-volume
Python pipeline for brain slices registration and volume reconstruction.

## Overview
Input datasets consist of images of mouse brain slices mounted sequentially after histological procedures. 
Large field of view images are obtained at two different resolutions.
- Lower resolution images are used for regsitration purposes.
- Higher resoltion images are used for a detailed analysis of area of interest.

Pipeline is aimed at perfoming registration on lower resolution *zoomed-out* images and applying transormations to higer resolution *zoomed-in* datasets.




https://github.com/user-attachments/assets/5736e3dd-4d64-40ea-83fe-7cd05b7d4b87





Overalapping region of transformed *zoomed-in* images is obtained and a volumetric reconstruction generated.



https://github.com/user-attachments/assets/69e6f269-2388-4817-9e22-6565464c8976




## Installation
Instructions for downloading relevant files and creating a python environment containing the required packages to the code.

1. [Install mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) environment manager.
2. [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) *slices-to-volume* repository.
3. Using a terminal navigate to the directory containing the repository and create a new environment using the file *slices_to_vol_env.yml*,
   ```
   # Run this line in the folder containing slices_to_vol_env.yml file
   mamba env create -f slices_to_vol_env.yml 
   ``` 
   If everything is correct an environment named *slices_to_vol_env* should be listed as the output of running `mamba env list` command.
   Check the [mamba user guide](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#mamba) for more information on how to use the environment manager.   
4. Activate environment, 
   ```
   mamba activate slices_to_vol_env
   ```
   and install kernel to be used with jupyter lab, naming it `slices_to_vol_k`
   ```
   ipython kernel install --user --name='slices_to_vol_k'
   ```
   
## Test notebook
1. Open a terminal and activate *slices_to_vol_env* environment.
   ```
   mamba activate slices_to_vol_env
   ``` 
2. Launch a jupyter lab instance.
   ```
   jupyter lab
   ```
3. Navigate to folder containing repository and open the *slice_to_volumen.ipynb* notebook.
4. Select the *slices_to_vol_k* kernel (*Kernel/Change Kernel...* in toolbar).
5. Follow the steps detailed on the notebook.

Script was tested on a laptop running on Pop!_OS 22.04 with 32GB of RAM memory and a 11th Gen Intel® Core™ i7 processor.

