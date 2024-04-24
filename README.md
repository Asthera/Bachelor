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

| N | Mean F1 Score        |
|---|----------------------|
| 1 | 0.3983488028564471   |
| 2 | 0.19309099081069694  |
| 3 | 0.2828702176088404   |
| 4 | 0.18923495241626498  |
| 5 | 0.18013008477871972  |
| 6 | ,0.30354764735342454 |
| 7 | 0.1974380579349123   |
| 8 | 0.27780086796118225  |
| 9 | 0.29636829738599646  |
| 10| 0.24985674537524435  |
| 11| 0.3116045607034342   |





