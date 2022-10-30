import argparse
from pyInfinityFlow.fcs_io import list_fcs_channels

def parse_boolean_string(input_string):
    if input_string == "True":
        return(True)
    elif input_string == "False":
        return(False)
    else:
        raise ValueError(f"Expected True or False, got {input_string}")

def main():
    parser = argparse.ArgumentParser(description='User Defined Arguments')
    parser.add_argument('--fcs_file', dest='fcs_file', type=str, required=True,
        help='FCS file from which to list channels. This is useful for writing '\
            'the name for a given channel in the backbone_annotation and '\
            'InfinityMarker_annotation files.')
    parser.add_argument('--add_user_defined_names', dest='add_user_defined_names', 
        type=str,
        help='Whether or not to include the user defined names which come from the '\
            '$PnS property in the TEXT segment values of the fcs file.',
        default="False")
    args = parser.parse_args()
    try:
        fcs_file = args.fcs_file
        add_u_names = parse_boolean_string(args.add_user_defined_names)
    except Exception as e:
        print(str(e))
        print("Failed to parse arguments in pyInfinityFlow-list_channels...")

    try:
        list_fcs_channels(fcs_file_path=fcs_file, 
            add_user_defined_names=add_u_names)
    except Exception as e:
        print(str(e))
        raise ValueError("Failed to read FCS files and list channel names.")


if __name__ == "__main__":
    main()