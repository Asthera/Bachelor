## in this file, we will check is all trasnforms names are writed correctly and it importes
import yaml
from transforms.builder import TransformsBuilder


def main():
    sweep_file_path = "sweep_paper_11_gpu.yaml"

    with open(sweep_file_path) as file:
        sweep = yaml.load(file, Loader=yaml.FullLoader)

    print("Checking yaml file...")

    if sweep['parameters']['yaml_file']['values'][0] == sweep_file_path:
        print("Yaml file path is correct")

    else:
        print("Error: Yaml file path is incorrect")
        print("In your sweep file, the path is:", sweep['parameters']['yaml_file']['values'][0])
        print("But it should be:", sweep_file_path)
        return

    print("Checking transforms...")

    transforms_from_yaml = sweep['parameters']['transform']['values']

    for i in transforms_from_yaml:
        try:
            TransformsBuilder(i).build()
        except:
            print(f"Error: Transform {i} not found in torchvision or monai, or custom transforms.")
            print("Please check the transform name in the yaml file")
            return

    print("All transforms are correct")


if __name__ == "__main__":
    main()
