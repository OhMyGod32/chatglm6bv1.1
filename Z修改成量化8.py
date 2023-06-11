import os

def replace_text(file_path, search_str, replace_str):
    with open(file_path, 'r') as file:
        content = file.read()

    content = content.replace(search_str, replace_str)

    with open(file_path, 'w') as file:
        file.write(content)

file_list = ['api.py', 'cli_demo.py', 'web_demo.py', 'web_demo2.py']
search_str = '.half().cuda()'
replace_str = '.quantize(8).half().cuda()'

for file_name in file_list:
    file_path = os.path.join(os.getcwd(), file_name)

    if os.path.exists(file_path):
        replace_text(file_path, search_str, replace_str)
        print(f'Text replacement complete for {file_name}.')
    else:
        print(f'{file_name} does not exist in the current directory.')