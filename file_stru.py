import os

def print_file_structure(start_path='.', indent=''):
    for item in os.listdir(start_path):
        item_path = os.path.join(start_path, item)
        print(indent + '|-- ' + item)
        if os.path.isdir(item_path):
            print_file_structure(item_path, indent + '    ')

# Example usage:
print_file_structure('.')  # You can change '.' to any directory path