import ast
import re
import yaml


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


def update_yaml_transforms(yaml_file_path, new_transform_values):
    # Read the existing YAML configuration
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the 'transform' values under 'parameters'
    if 'parameters' in config and 'transform' in config['parameters']:
        config['parameters']['transform']['values'] = new_transform_values
    else:
        # If the structure is not as expected, create or recreate the necessary structure
        if 'parameters' not in config:
            config['parameters'] = {}
        config['parameters']['transform'] = {'values': new_transform_values}

    # Write the updated configuration back to the YAML file
    with open(yaml_file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


# Assume the file 'script.py' contains the definitions of the transformations
transforms = read_transforms('sweep_paper_11_gpu.py')
update_yaml_transforms('sweep_paper_11_600x400_gpu.yaml', transforms)