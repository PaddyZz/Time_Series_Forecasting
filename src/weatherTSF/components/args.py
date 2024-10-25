import yaml
import argparse
def cast_value(value):

    if value.isdigit():
        return int(value)
    

    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    

    return value.strip().strip("'\"")
def args_cope():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, nargs='+')
    args = parser.parse_args()
    if not args.config:
        return

    with open('params.yaml', 'r') as file:
        config = yaml.safe_load(file)


    for param in args.config:
        key, value = param.split('=', 1)
        key = key.strip()
        new_value = cast_value(value.strip())  
        if key.strip().upper() in config:  
            config[key.strip().upper()] = new_value   
       


    with open('params.yaml', 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)
