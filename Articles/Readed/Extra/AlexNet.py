import torchvision.transforms as transforms

# Define the transform
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256 pixels
    transforms.RandomResizedCrop(224),  # Randomly crop a 224x224 patch
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally with a default probability of 0.5
    # Add any other transforms here (e.g., ToTensor, normalization)
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
])

