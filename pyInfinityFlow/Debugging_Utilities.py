"""
Print a list to the console, one element per line with no quotations.
"""
def pprint(input_list):
    print("\n".join(input_list))


"""
Print statements to the console, allowing different DEBUGGING verbosity levels.
"""
def printv(verbosity, prefix_debug=True, v1 = "", v2 = "", v3 = ""):
    if prefix_debug:
        debug_prefix = "DEBUG: "
    else:
        debug_prefix = ""
    v1_end = "" if v1 == "" else "\n"
    v2_end = "" if v2 == "" else "\n"
    v3_end = "" if v3 == "" else "\n"
    if verbosity == 3:
        if len(v1) > 0: print(v1, end=v1_end)
        if len(v2) > 0: print(v2, end=v2_end)
        if len(v3) > 0: print(debug_prefix + v3, end=v3_end)
    elif verbosity == 2:
        if len(v1) > 0: print(v1, end=v1_end)
        if len(v2) > 0: print(v2, end=v2_end)
    elif verbosity == 1:
        if len(v1) > 0: print(v1, end=v1_end)