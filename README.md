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
3. Maybe implement deterministic seeds ? - Added
4. Try in output transform resize to format that take ImageNet pretrained models
5. Try without pretraining
6. Try with dropout
7. Add more metrics (good discussed in "Medical image data augmentation: techniques, comparisons and interpretations") [MCC, accuraccy, kappa ]


# Tested
1. From paper "Medical image data augmentation: techniques, comparisons and interpretations.pdf"
Was tested with 11 augmentation methods, like it was tested in the paper.
Methods:
   1. Shear  [−15°, 15°], prob=0.5
   2. Translate [0.15, 0.15], y-axis and x-axis
   3. Rotate  [-25°, 25°]
   4. Add Gaussian noise with fixed variance [0.3, 0.4, 0.5]
   5. Add salt and pepper noise with density [0.01, 0.02, 0.03]
   6. Salt and pepper noise with density [0.01, 0.02, 0.03] ans Shear  [−15°, 15°]
   7. Add Gaussian noise with fixed variance [0.3, 0.4, 0.5] and Rotate  [-25°, 25°]
   8. Rotate  [-25°, 25°] and Translate [0.15, 0.15], y-axis and x-axis
   9. Translate [0.15, 0.15], y-axis and x-axis and Shear  [−15°, 15°]
   10. Translate [0.15, 0.15] y-axis and x-axis, shear  [-15°, 15°] and Rotate  [-25°, 25°]
   11. Color shifting, sharpening, and contrast

Was tested with init transforms 600x400 or 600x600, 600x400 better (`CrossValidation/utills_sweep/analyze_sweep_paper_11.ipynb`)

Implementation of them is in `CrossValidation/Transforms.py` and `CrossValidation/sweep_paper_11_gpu.py` <br/>
Results are in `CrossValidation/utills_sweep/sweeps_edited or sweeps_raw/25oanqcs.csv`

| Transforms | Test F1_mean | Test F1_max | Test F1_min | Test Precision_mean | Test Precision_max | Test Precision_min | Test Recall_mean | Test Recall_max | Test Recall_min |
|------------|--------------|-------------|-------------|---------------------|--------------------|--------------------|------------------|-----------------|-----------------|
| No Transforms | 0.313 | 0.825 | 0.000 | 0.560 | 1.000 | 0.000 | 0.239 | 0.702 | 0.000 |
| Color shifting, sharpening, and contrast | 0.312 | 0.840 | 0.054 | 0.472 | 1.000 | 0.035 | 0.393 | 0.724 | 0.111 |
| Rotate and Translate | 0.278 | 0.824 | 0.057 | 0.455 | 1.000 | 0.038 | 0.402 | 0.862 | 0.037 |
| Rotate | 0.283 | 0.699 | 0.046 | 0.312 | 0.643 | 0.029 | 0.408 | 0.766 | 0.111 |
| Shear | 0.398 | 0.824 | 0.067 | 0.468 | 1.000 | 0.048 | 0.410 | 0.745 | 0.111 |
| Translate, Shear, and Rotate | 0.250 | 0.642 | 0.075 | 0.352 | 1.000 | 0.057 | 0.340 | 0.745 | 0.111 |
| Translate and Shear | 0.296 | 0.625 | 0.111 | 0.369 | 1.000 | 0.076 | 0.376 | 0.745 | 0.148 |
| Translate | 0.193 | 0.496 | 0.049 | 0.311 | 1.000 | 0.032 | 0.287 | 0.745 | 0.069 |
| Gaussian noise and Rotate | 0.197 | 0.554 | 0.000 | 0.327 | 1.000 | 0.000 | 0.274 | 0.872 | 0.000 |
| Gaussian noise | 0.189 | 0.753 | 0.000 | 0.194 | 0.761 | 0.000 | 0.194 | 0.745 | 0.000 |
| Salt and Pepper noise and Shear | 0.304 | 0.650 | 0.047 | 0.320 | 1.000 | 0.030 | 0.473 | 0.828 | 0.111 |
| Salt and Pepper noise | 0.180 | 0.761 | 0.000 | 0.177 | 0.778 | 0.000 | 0.196 | 0.745 | 0.000 |





