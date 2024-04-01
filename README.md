# Bachelor 

This is the repository for my Bachelor's thesis.
Title: "Implementation of augmentation methods for medical ultrasound data"

Temporary working directory: "/CrossValidation"
Articles: "/Articles"

Project at [WandB](https://wandb.ai/daswoldemar/bachelor?nw=nwuserdaswoldemar)




## Instructions from the University

1. Create an overview of image data augmentation associated with neural network training with a focus on medical image data.
2. Provide an overview of existing solutions for parallel hyperparameter tuning in neural networks.
3. Use the processed knowledge to design augmentation methods suitable for ultrasound image data.
4. Apply the selected methods and develop a mechanism for automatic and parallel validation of the proposed experiments.
5. Compare the results of the different augmentation techniques and perform an appropriate evaluation.
6. Prepare documentation as directed by the thesis advisor.


## Practical Side

In theory, researcher often not say which augmentation methods they used. They just say that they used augmentation. 
without parameters intervals 

### To Do

1. Experiment with Mixup, Cutout, Cutmix
2. Maybe try to implement AutoAugment
3. Maybe implement deterministic seeds ?
4. Try in output transform resize to format that take ImageNet pretrained models
5. Try without pretraining
6. Try with dropout
7. Add more metrics (good discussed in "Medical image data augmentation: techniques, comparisons and interpretations")

Strategies
1. Affine transformations
   1. From "An Integration of blockchain and AI ..."
       1. 