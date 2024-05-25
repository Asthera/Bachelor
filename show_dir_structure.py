import os


def print_folder_structure(folder_path, indent_level=0):
    try:
        items = os.listdir(folder_path)
    except PermissionError:
        # If the folder cannot be accessed, print an error message and return
        print("  " * indent_level + "PermissionError: Unable to access " + folder_path)
        return

    for item in items:
        item_path = os.path.join(folder_path, item)
        print("  " * indent_level + "|-- " + item)
        if os.path.isdir(item_path):
            print_folder_structure(item_path, indent_level + 1)


# Replace 'your_folder_path' with the path to your folder
your_folder_path = "Practical"
print(your_folder_path)
print_folder_structure(your_folder_path)
