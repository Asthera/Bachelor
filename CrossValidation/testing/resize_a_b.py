import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw

# Define the resize transformation
resize = T.Resize((500, 200))

# Read the images
image_a = read_image('withAlineNoSliding.png') / 255.0
image_b = read_image("withBline.png") / 255.0

# make 3 channels
image_a = image_a.expand(3, -1, -1)
image_b = image_b.expand(3, -1, -1)

# Resize the images
image_a = resize(image_a)
image_b = resize(image_b)

# Convert tensors to PIL images
image_a_pil = to_pil_image(image_a)
image_b_pil = to_pil_image(image_b)

# Draw bounding boxes
draw_a = ImageDraw.Draw(image_a_pil)
draw_b = ImageDraw.Draw(image_b_pil)

# For image_a, draw a red line between points (45, 224) and (157, 244)
draw_a.rectangle([46, 262, 160, 277], outline="red", width=2)

# For image_b, draw a blue line between points (146, 193) and (193, 417)
draw_b.rectangle([152, 187, 180, 385], outline="blue", width=2)

# Save the images with bounding boxes
image_a_pil.save('resized_withAlineNoSliding_with_bounding_box.png')
image_b_pil.save('resized_withBline_with_bounding_box.png')
