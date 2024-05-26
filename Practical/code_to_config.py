import ast
import re
import yaml
import os
import shutil


def read_transforms(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Parse the content of the file into an AST
    tree = ast.parse(content)

    # This dictionary will hold the transformation settings
    transforms = {}

    transforms_values = []

    # Walk through the AST nodes
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            # Check if the variable name starts with 'transform_'
            name = node.targets[0].id
            if name.startswith('transform_'):
                # Check if the right-hand side is a List
                if isinstance(node.value, ast.List):
                    # Extract each element in the list separately
                    transform_list = []
                    for element in node.value.elts:
                        element_code = ast.unparse(element)
                        element_code = re.sub(r'\s+', ' ', element_code).strip()
                        transform_list.append(element_code)
                    transforms[name] = transform_list

    # Print the transforms in the required format
    for key in transforms.keys():
        t = []
        for transform in transforms[key]:
            t.append(transform)

        transforms_values.append(t)

    return transforms_values


def update_yaml_transforms(transforms, base_yaml_path, new_yaml_path):
    new_yaml = new_yaml_path[:-3] + ".yaml"
    # change configs_code/sweep_combined_f-1_mean.py to configs/sweep_combined_f-1_mean.yaml

    new_yaml = new_yaml.replace('configs_code', 'configs')



    # Check if the new YAML file exists
    if not os.path.exists(new_yaml):
        # Copy the base YAML file to a new file if it does not exist
        shutil.copy(base_yaml_file_path, new_yaml)

    # Read the existing YAML configuration
    with open(new_yaml, 'r') as file:
        config = yaml.safe_load(file)


    # Update the 'transform' values under 'parameters'
    if 'parameters' in config and 'transform' in config['parameters']:
        config['parameters']['transform']['values'] = transforms
        config['parameters']['yaml_file']['values'] = [new_yaml]
    else:
        # If the structure is not as expected, create or recreate the necessary structure
        if 'parameters' not in config:
            config['parameters'] = {}
        config['parameters']['transform'] = {'values': transforms}

    # Write the updated configuration back to the YAML file
    with open(new_yaml, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


base_yaml_file_path = 'configs/sweep_gpu.yaml'
file_with_transforms = 'configs_code/sweep_combined_f-1_mean.py'
# Assume the file 'script.py' contains the definitions of the transformations
transforms = read_transforms(file_with_transforms)
update_yaml_transforms(transforms, base_yaml_file_path, file_with_transforms)
