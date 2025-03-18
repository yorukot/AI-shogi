import os


def modify_yolo_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('2'):
            modified_lines.append('0' + line[1:])
        elif line.startswith('3'):
            modified_lines.append('1' + line[1:])
        else:
            modified_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(modified_lines)


def find_and_modify_txt_files():
    current_dir = os.getcwd()
    for file_name in os.listdir(current_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(current_dir, file_name)
            modify_yolo_labels(file_path)
            print(f'Modified: {file_name}')


if __name__ == "__main__":
    find_and_modify_txt_files()
