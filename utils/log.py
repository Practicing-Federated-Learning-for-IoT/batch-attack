import json

def write_info_to_txt(info,name):
    f_name = name
    with open("{}.txt".format(name), 'w') as f:
        json.dump(info,f)