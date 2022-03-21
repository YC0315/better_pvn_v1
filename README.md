
## Introduction
We reiterate that our work is based on pvnet. we provide pvnet here for reference(https://github.com/zju3dv/clean-pvnet), and thanks again to Peng et al. for their excellent work.

We made the following change：  
- [x]  After the network generates the vector field, when calculating candidate key points based on the vectors on the same object, we first align and filter to prevent the deviation from being too small, resulting in the generation of hypotheses with too large deviation. The filtering angle is different for each class on the two datasets, so this threshold needs to be set according to the class.


## Training and Testing Again
The code for installation, network training, and testing can be found in the PVNet link in the introduction section. The training and testing commands are shown below:<br>

### Take the benchvise as an example
1. Prepare the data related to `benchvise`:<br>
    
    >python run.py --type linemod cls_type benchvise
    
2. train:<br>

    >python train_net.py --cfg_file configs/linemod.yaml model yourmodel_dir cls_type benchvise
    
      
3. test with the uncertainty-driven PnP on Linemod and OCC-Linemod datasets:<br>

    >export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib/csrc/uncertainty_pnp/lib<br>
    python run.py --type evaluate --cfg_file configs/linemod.yaml model yourmodel_dir cls_type benchvise test.un_pnp True<br>
    python run.py --type evaluate --cfg_file configs/linemod.yaml test.dataset LinemodOccTest model yourmodel_dir cls_type benchvise test.un_pnp True
    
   
## Schematic
![Schematic](https://github.com/YC0315/better_pvn_v1/blob/da189cb63a27f56a43a107d27fb6e42b4206bb18/design%20sketch/Schematic.png)



