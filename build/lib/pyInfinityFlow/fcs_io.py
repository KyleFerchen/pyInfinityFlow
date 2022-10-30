import pandas as pd
import numpy as np
import re
from struct import unpack
import datetime

### Required parameters in FCS3.1 TEXT segment keyword-value pairs
SUPPORTED_VERSIONS = ['FCS3.1']
FCS_TEXT_SEGMENT_SIZE_LIMIT = 99000000
REQUIRED_FCS_3_1 = np.array(['$BEGINANALYSIS', '$BEGINDATA', '$BEGINSTEXT', '$BYTEORD',
                                '$DATATYPE', '$ENDANALYSIS', '$ENDDATA', '$ENDSTEXT',
                                '$MODE', '$NEXTDATA', '$PAR', '$TOT'])
FREQUENT_LINEAR_CHANNELS = np.array(['FSC-W', 'FSC-H', 'FSC-A', 'SSC-B-W', 'SSC-B-H', 'SSC-B-A'
                                     'SSC-W', 'SSC-H', 'SSC-A', 'umap-x', 'umap-y'])


class FCSFileObject:
    """Primary class for working with FCS files.

    This class is used to read and write FCS files. A mode is specified to either 
    read from or write to the given fcs_file_path. Reading of FCS files can be
    done without including the DATA segment, so that the HEADER and TEXT segments 
    can be read quickly.

    Warning
    -------
    Currently only FCS3.1 files are supported.

    Arguments
    ----------
    fcs_file_path : str
        The path to the FCS file. (Required)
    mode : str (Epects 'r'|'w')
        The mode in which to treat the FCS file. If 'r', the class instance will \
        read from the FCS file immediately after it is created. (Default='r')
    read_data_segment : bool
        Whether or not to read in the DATA segment of the FCS file. If false, \
        this allows you to read in the HEADER and TEXT segment values into the \
        class to learn important properties from the FCS file (Eg. The number \
        of events captured, the channel names, etc.) (Default=True)

    Attributes
    ----------
    file_path : str
        The path to the FCS file. Set by fcs_file_path
    byte_locations : dict{KEY: int}
        The binary positions of the files marking the different segments. This \
        will be filled when mode='r' upon instantiation with the following keys:
            - ["text_start"]
            - ["text_end"]
            - ["data_start"]
            - ["data_end"]
            - ["analysis_start"]
            - ["analysis_end"]
    version : str
        The version of the .fcs file (Eg. 'FCS3.1')
    text_segment : str
        The TEXT segment as a string.
    delimiter : str
        This is the character used as the delimiter between items in the TEXT \
        segment
    byteord_format : str
        The byte order format to use to read the DATA segment
    text_segment_values : dict{KEY: str}
        A dictionary that stores the FCS file TEXT segment key-value entries. \
        These are important for defining properties about the channels, file \
        positions, experiment annotations, etc.
    spillover : pandas.DataFrame
        The spillover matrix to use for compensation
    data : pandas.DataFrame
        The data from the DATA segment of the FCS file
    struct_format_string : str
        Struct format string to pack and unpack the DATA segment as binary
    par_count : int
        The number of parameters in the FCSFileObject
    list_par_n : list[int]
        Ordered list of parameters by number
    named_par : list[str]
        The $PnS names, usually defined by the user when the FCS data is \
        captured
    named_par_channel : list[str]
        The $PnN channel names, these must be unique and are generally defined \
        by the software used to capture the FCS events (Eg. "GFP-A")
    """


    def __init__(self, fcs_file_path="", mode='r', read_data_segment=True):
        self.file_path = fcs_file_path      # The file path, typically input when instantiated
        self.byte_locations = {}            # Byte locations to read/write for HEADER segment
        self.version = ""                   # Version of .fcs file (Eg. 'FCS3.1')
        self.text_segment = ""              # The TEXT segment of the .fcs file (UTF-8 encoded)
        self.delimiter = chr(13)            # Delimiter to use between items in the TEXT segment
        self.byteord_format = ""            # The byte order format to use to read the DATA segment
        self.text_segment_values = {}       # Dictionary of keyword-value pairs from TEXT segment
        self.spillover = pd.DataFrame([])   # Spillover matrix from TEXT segment
        self.data = pd.DataFrame([])        # Data from DATA segment
        self.struct_format_string = ""      # Struct format string to pack and unpack the DATA segment as binary
        self.par_count = 0                  # The number of parameters in the FCSFileObject
        self.list_par_n = []                # Ordered list of parameters by number
        self.named_par = []                 # $PnS names, usually user defined
        self.named_par_channel = []         # Raw $PnN required names, following self.list_par_n order
        # Use ASCII 11 as the preliminary substitute for the delimiter
        self.track_delimiter = chr(11)
        ### Read in the file header and text segment ###
        if mode == 'r' or mode == 'R':
            try:
                with open(self.file_path, 'rb') as tmp_file:
                    #### Reading in the header information
                    # The version identifier v_id, tells in which version of .fcs file the current file is written (Eg. FCS3.1)
                    self.version = read_binary_position(tmp_file, 0, 6).replace(" ", "")
                    # The start of the primary TEXT segment.
                    self.byte_locations["text_start"] = read_binary_position(tmp_file, 10, 17, output_type='int')
                    # The last byte of the primary TEXT segment.
                    self.byte_locations["text_end"] = read_binary_position(tmp_file, 18, 25, output_type='int')
                    # The byte offset for the start of the DATA segment
                    self.byte_locations["data_start"] = read_binary_position(tmp_file, 26, 33, output_type='int')
                    # The byte offset for the end of the DATA segment
                    self.byte_locations["data_end"] = read_binary_position(tmp_file, 34, 41, output_type='int')
                    # The byte offset for the start of the ANALYSIS segment
                    self.byte_locations["analysis_start"] = read_binary_position(tmp_file, 42, 49, output_type='int')
                    # The byte offset for the end of the ANALYSIS segment
                    self.byte_locations["analysis_end"] = read_binary_position(tmp_file, 50, 57, output_type='int')
                    #### Reading in the TEXT segment information
                    self.text_segment = read_binary_position(tmp_file, self.byte_locations["text_start"], self.byte_locations["text_end"])
            except:
                raise ValueError("Error! Unable to read fcs header for {}...".format(self.file_path))
            ### Process the primary text segment ###
            try:
                # Get the delimiter and have a track_delimiter to replace occurrences from keyword-value pairs in TEXT
                self.delimiter = self.text_segment[0]
                if self.delimiter == self.track_delimiter:
                    self.track_delimiter = chr(12)
                # Parse the primary TEXT segment, filling self.text_segment_values with keyword-value pairs
                self.text_segment = self.text_segment.replace("{}{}".format(self.delimiter, self.delimiter), self.track_delimiter)
                tmp_text_array = self.text_segment.split(self.delimiter)[1:-1]
                if len(tmp_text_array) % 2 == 0:
                    for i in range(int(len(tmp_text_array) / 2)):
                        # Make the keyword uppercase (keywords are case insensitive)
                        tmp_keyword = tmp_text_array[(2*i)].replace(self.track_delimiter, self.delimiter).upper()
                        tmp_value = tmp_text_array[((2*i) + 1)].replace(self.track_delimiter, self.delimiter).replace(" ", "")
                        self.text_segment_values[tmp_keyword] = tmp_value
                else:
                    raise ValueError("Error! Unequal keyword-value pairs in TEXT segment for {}...".format(self.file_path))
                ### Check text_segment_values for required parameters ###
                tmp_check_required_params = np.isin(REQUIRED_FCS_3_1, list(self.text_segment_values.keys()))
                if sum(tmp_check_required_params) != len(REQUIRED_FCS_3_1):
                    raise ValueError("Error! Required params not present!")
                ### Check all event parameters are present ###
                self.par_count = int(self.text_segment_values["$PAR"])
                ### Set up regex to find parameter values
                delimiter_escaped = re.escape(self.delimiter)
                FACS_PARAMETER_B_REGEX = re.compile(f'{delimiter_escaped}\\$P([0-9]+)B{delimiter_escaped}')
                FACS_PARAMETER_E_REGEX = re.compile(f'{delimiter_escaped}\\$P([0-9]+)E{delimiter_escaped}')
                FACS_PARAMETER_N_REGEX = re.compile(f'{delimiter_escaped}\\$P([0-9]+)N{delimiter_escaped}')
                FACS_PARAMETER_R_REGEX = re.compile(f'{delimiter_escaped}\\$P([0-9]+)R{delimiter_escaped}')
                tmp_param_b = sorted(np.array(FACS_PARAMETER_B_REGEX.findall(self.text_segment), dtype=int))
                tmp_param_e = sorted(np.array(FACS_PARAMETER_E_REGEX.findall(self.text_segment), dtype=int))
                tmp_param_n = sorted(np.array(FACS_PARAMETER_N_REGEX.findall(self.text_segment), dtype=int))
                tmp_param_r = sorted(np.array(FACS_PARAMETER_R_REGEX.findall(self.text_segment), dtype=int))                
                if tmp_param_b == tmp_param_e == tmp_param_n == tmp_param_r:
                    tmp_len_input_par = len(tmp_param_b)
                    if tmp_len_input_par == self.par_count:
                        self.list_par_n = tmp_param_b
                    else:
                        raise ValueError("Error! {} parameters found. Expected {}.".format(tmp_len_input_par, self.text_segment_values))
                else:
                    raise ValueError("Error! $PnB, $PnE, $PnN, and $PnR values don't match.")
                ### Check to see if USER defined names are included ###
                # USER defined names should be in $PnS
                tmp_par_is_named = np.array(["$P{}S".format(i) in self.text_segment_values for i in self.list_par_n])
                tmp_list_par = np.array([self.text_segment_values["$P{}N".format(i)] for i in self.list_par_n])
                if sum(tmp_par_is_named) > 0:
                    tmp_par_names = np.array([self.text_segment_values["$P{}S".format(i)] if tmp_mask else "" for i, tmp_mask in zip(self.list_par_n, tmp_par_is_named)])
                else:
                    tmp_par_names = np.array([""] * self.par_count)

                self.named_par = tmp_par_names
                self.named_par_channel = tmp_list_par
                ### Read in the spillover matrix ###
                if "$SPILLOVER" in self.text_segment_values:
                    try:
                        tmp_spillover = self.text_segment_values["$SPILLOVER"].split(",")
                        tmp_spillover_n = int(tmp_spillover[0])
                        tmp_spillover_names = tmp_spillover[1:(tmp_spillover_n + 1)]
                        # Make sure the spillover names are present in the channel names
                        tmp_check_spillover_names = np.isin(tmp_spillover_names, self.named_par_channel)
                        if sum(tmp_check_spillover_names) != len(tmp_spillover_names):
                            raise ValueError("Error! Channel names in spillover matrix don't match '$PnN' from text segment for file {}.", self.file_path)

                        self.spillover = pd.DataFrame(np.reshape(np.array(tmp_spillover[(tmp_spillover_n + 1):],
                                                                            dtype=float),
                                                                    newshape = (tmp_spillover_n, tmp_spillover_n),
                                                                    order = 'C'),
                                                        index = tmp_spillover_names,
                                                        columns = tmp_spillover_names)
                    except:
                        raise ValueError("Error! Couldn't process $SPILLOVER value!")
                ### Generate the format string to read in the DATA segment ###
                # Only $MODE = 'L" is currently supported, as others are deprecated
                if self.text_segment_values["$MODE"] == 'L':
                    # Detect the byte order: 1,2,3,4 == little endian | 4,3,2,1 == big endian
                    tmp_endian_code = self.text_segment_values["$BYTEORD"]
                    if tmp_endian_code == "1,2,3,4":
                        self.byteord_format = "<"
                    elif tmp_endian_code == "4,3,2,1":
                        self.byteord_format = ">"
                    else:
                        raise ValueError("$BYTEORD value not recognized: {}.".format(tmp_endian_code))

                    tmp_n_tot_values = self.par_count * int(self.text_segment_values["$TOT"])
                    if self.text_segment_values["$DATATYPE"] == 'F':
                        self.struct_format_string = '{}{}f'.format(self.byteord_format, tmp_n_tot_values)
                    elif self.text_segment_values["$DATATYPE"] == 'D':
                        self.struct_format_string = '{}{}d'.format(self.byteord_format, tmp_n_tot_values)
                    else:
                        raise ValueError("ERROR! Only $DATATYPE values 'F' and 'D' are supported.")
                else:
                    raise ValueError("ERROR! $MODE {} detected. Only 'L' is supported.".format(self.text_segment_values["$MODE"]))
            except:
                raise ValueError("Error! Unable to process TEXT for {}...".format(self.file_path))
            ### Process the DATA segment ###
            if read_data_segment:
                try:
                    self.data = self._read_data()
                except:
                    raise ValueError("Error! Unable to process DATA for {}...".format(self.file_path))

    def _read_data(self):
        ### Read in the Data Matrix after the class instance is instantiated ###
        try:
            with open(self.file_path, "rb") as tmp_file:
                if self.text_segment_values["$BEGINDATA"] != "0":
                    tmp_seek_position = int(self.text_segment_values["$BEGINDATA"])
                    tmp_n_bytes = 1 + int(self.text_segment_values["$ENDDATA"]) - int(self.text_segment_values["$BEGINDATA"])
                else:
                    tmp_seek_position = int(self.byte_locations["data_start"])
                    tmp_n_bytes = 1 + int(self.byte_locations["data_end"]) - int(self.byte_locations["data_start"])

                tmp_file.seek(int(tmp_seek_position))
                tmp_channel_names = self.named_par_channel
                return(pd.DataFrame(np.reshape(unpack(self.struct_format_string, tmp_file.read(tmp_n_bytes)),
                                                newshape = (int(self.text_segment_values["$TOT"]),
                                                            self.par_count),
                                                order='C'),
                                    columns = tmp_channel_names))
        except:
            raise ValueError("Error! Failed to read the DATA segment!")

    def load_data_from_pd_df(self, input_pd_df, input_channel_names=[], 
            input_spillover_matrix=None, additional_text_segment_values={}):
        """Load data from a pandas.DataFrame into an FCSFileObject

        This method allows the FCSFileObject to add data to the DATA segment 
        from a pandas.DataFrame. The resulting data can then be written to an
        FCS file using the FCSFileObject.to_fcs() method

        Parameters
        ----------
        input_pd_df : pandas.DataFrame
            The DATA to add to the DATA segment. (Required)
        input_channel_names : list[str]
            These are the names to give to the $PnS TEXT segment key-value. \
            The list must be the same size as input_pd_df.shape[1] (The columns \
            of the input_pd_df) or be empty (Default=[])
        input_spillover_matrix : None or pandas.DataFrame
            The spillover matrix for fluorescence compensation (Default=None)
        additional_text_segment_values : dict{'str':'str'}
            Key-value pairs to add to the TEXT segment of the FCS file. \
            (Default={})

        Returns
        -------
        None
            Adds the given information to the FCSFileObject

        """
        ### Check input values ###
        if len(input_pd_df.shape) != 2: raise ValueError("Error! input_pd_df must be a 2-dimensional pandas dataframe.")
        if input_pd_df.shape[1] < 1: raise ValueError("Error! input_pd_df must have at least one channel feature.")
        if input_pd_df.shape[0] < 1: raise ValueError("Error! input_pd_df must have at least one event.")
        tmp_par_count = input_pd_df.shape[1]
        # Check parameter naming values
        tmp_channel_names = list(input_pd_df.columns.values)
        tmp_len_user_channel_names = len(input_channel_names)
        tmp_user_channel_names_present = False
        if tmp_len_user_channel_names > 0:
            tmp_user_channel_names_present = True
            if tmp_len_user_channel_names != len(tmp_channel_names): 
                raise ValueError("Error! input_pd_df.shape[1] and len(input_channel_names) must match, and in the same order. Fill <input_channel_names> with empty strings if necessary.")

            tmp_rename_channels_ordered = input_channel_names
        
        else:
            tmp_rename_channels_ordered = tmp_channel_names

        # Check values to convert to double
        try:
            input_pd_df.values.astype(float)
        except:
            raise ValueError("Error! Input dataframe values could not be typecast to float.")

        # Check input_spillover_matrix
        if input_spillover_matrix is None:
            print("Omitting spillover matrix...")
        else:
            if len(input_spillover_matrix.shape) != 2: raise ValueError("Error! input_pd_df must be a 2-dimensional pandas dataframe.")
            if input_spillover_matrix.shape[0] > 0:
                if input_spillover_matrix.shape[0] != input_spillover_matrix.shape[1]: raise ValueError("Error! input_spillover_matrix must be an n x n Pandas DataFrame.")
                tmp_spillover_col_names = list(input_spillover_matrix.columns.values)
                tmp_spillover_row_names = list(input_spillover_matrix.index.values)
                if tmp_spillover_col_names != tmp_spillover_row_names: raise ValueError("Error! input_spillover_matrix row and col names must be identical.")
                tmp_check_spillover_names = np.isin(tmp_spillover_col_names, tmp_channel_names)
                if sum(tmp_check_spillover_names) != len(tmp_spillover_col_names):
                    raise ValueError("Error! input_spillover_matrix featuere names must be in input_pd_df.columns.values.", self.file_path)

            # If no valid spillover matrix is provided, use the identity matrix
            else:
                tmp_spillover_col_names = input_pd_df.columns.values
                tmp_spillover_col_names = np.setdiff1d(tmp_spillover_col_names, FREQUENT_LINEAR_CHANNELS)
                input_spillover_matrix = pd.DataFrame(np.identity(tmp_par_count),
                                                        index = tmp_spillover_col_names,
                                                        columns = tmp_spillover_col_names)

        ### Initialize FCSFileObject with pandas dataframe parameters ###
        self.par_count = tmp_par_count
        self.list_par_n = list(range(1, self.par_count + 1))
        self.named_par = input_channel_names
        self.named_par_channel = tmp_channel_names
        self.spillover = input_spillover_matrix
        self.data = input_pd_df
        try:
            self.text_segment_values.update(additional_text_segment_values)
        except:
            print("WARNING! Failed to add additional_text_segment_values.")

    def _prepare_required_fcs_properties(self, fcs_version="FCS3.1"):
        ### Set up parameter associated required properties ###
        # Check self.data
        if len(self.data.shape) != 2: raise ValueError("Error in prepare_required_fcs_properties()! self.data must be a 2-dimensional pandas dataframe.")
        if self.data.shape[1] < 1: raise ValueError("Error in prepare_required_fcs_properties()! self.data must have at least one channel feature.")
        if self.data.shape[0] < 1: raise ValueError("Error in prepare_required_fcs_properties()! self.data must have at least one event.")
        # Determine the bit size needed for the data
        tmp_data_max_value = self.data.max().max()
        if tmp_data_max_value >= 1.797693134862315E+308:
            raise ValueError("Error! Dataset has a max value that exceeds 1.797693134862315E+308, out of double precision floating point byte size range...")
        elif tmp_data_max_value > 3.402E38:
            tmp_datatype = "D"
        else:
            tmp_datatype = "F"

        # Handling parameter values
        if self.par_count == self.data.shape[1] == len(self.list_par_n) == len(self.named_par_channel) == len(self.named_par):
            tmp_par_text_value_segments = {}
            for i, tmp_n in enumerate(self.list_par_n):
                tmp_max_value = self.data.iloc[:,i].max()
                tmp_channel_name = self.named_par_channel[i]
                tmp_par_text_value_segments["$P{}B".format(tmp_n)] = '32' if tmp_datatype == "F" else '64'
                tmp_par_text_value_segments["$P{}E".format(tmp_n)] = '0,0'
                tmp_par_text_value_segments["$P{}N".format(tmp_n)] = tmp_channel_name
                tmp_par_text_value_segments["$P{}R".format(tmp_n)] = str(tmp_max_value)
                tmp_par_text_value_segments["$P{}S".format(tmp_n)] = self.named_par[i]
                if tmp_channel_name in FREQUENT_LINEAR_CHANNELS:
                    tmp_min_value = self.data.iloc[:,i].min()
                    tmp_5pct_adj = 0.05 * (tmp_max_value - tmp_min_value)
                    tmp_par_text_value_segments["$P{}D".format(tmp_n)] = "Linear,{:.2f},{:.2f}".format((tmp_min_value - tmp_5pct_adj), 
                                                                                                       (tmp_max_value + tmp_5pct_adj))
                # else:
                #     tmp_estimate_decades = math.ceil(tmp_max_value / 10)
                #     tmp_par_text_value_segments["$P{}D".format(tmp_n)] = "Logarithmic,{},0.1".format(tmp_estimate_decades)

            self.text_segment_values = dict(self.text_segment_values, **tmp_par_text_value_segments)

        else:
            tmp_error_string = "Error! Number of parameters do not align:\n\tpar_count: {}\n\tData: {}\n\tlist_par_n: {}\n\tnamed_par_channel: {}\n\tnamed_par: {}"
            raise ValueError(tmp_error_string.format(self.par_count, 
                                                     self.data.shape[1], 
                                                     len(self.list_par_n), 
                                                     len(self.named_par_channel), 
                                                     len(self.named_par)))
        
        ### Set required fcs properties in the class instance ###
        self.delimiter = chr(13)            # This program uses ASCII 3 ("CR") as the default delimiter
        self.byteord_format = "<"            # This program uses little endian as the default byte order
        ### Set the self.text_segment_values keyword-value pairs ###
        # Note, the following need to be added as the file is being written:
        # $BEGINANALYSIS $BEGINDATA $BEGINSTEXT $ENDANALYSIS $ENDDATA $ENDSTEXT
        self.text_segment_values["$DATATYPE"] = tmp_datatype
        self.text_segment_values["$MODE"] = "L"
        self.text_segment_values["$BYTEORD"] = "1,2,3,4"
        self.text_segment_values["$NEXTDATA"] = "0"
        self.text_segment_values["$PAR"] = str(self.data.shape[1])
        self.text_segment_values["$TOT"] = str(self.data.shape[0])
        self.struct_format_string = "<{}{}".format(self.data.shape[0] * self.data.shape[1], tmp_datatype.lower())
        self.version = "FCS3.1"
        # The following values will be computed when the file is written out
        self.text_segment_values['$BEGINANALYSIS'] = "0"
        self.text_segment_values['$BEGINDATA'] = "0"
        self.text_segment_values['$BEGINSTEXT'] = "0"
        self.text_segment_values['$ENDANALYSIS'] = "0"
        self.text_segment_values['$ENDDATA'] = "0"
        self.text_segment_values['$ENDSTEXT'] = "0"

    def _add_spillover_matrix_to_text_value_segment(self):
        if len(self.spillover.shape) != 2: raise ValueError("ERROR! Spillover matrix must be a 2 dimensional dataframe.")
        if self.spillover.shape[0] > 0:
            try:
                if self.spillover.shape[0] != self.spillover.shape[1]: raise ValueError("ERROR! Spillover matrix must be sqaure shaped.")
                tmp_feature_names = list(self.spillover.columns.values)
                self.text_segment_values["$SPILLOVER"] = "{},{},{}".format(self.spillover.shape[0],
                                                                            ",".join(tmp_feature_names),
                                                                            ",".join(list(self.spillover.astype('str').values.flatten())))
            except:
                raise ValueError("ERROR! Failed to add spillover matrix to self.text_segment_values.")

    def _check_text_segment_key(self, input_str):
        ascii_arr = np.array([ord(item) for item in input_str])
        ascii_check = ~((ascii_arr > 31) & (ascii_arr < 127))
        return(sum(ascii_check) == 0)

    def _write_fcs(self, fcs_file_path):
        ### Check Values ###
        if self.version not in SUPPORTED_VERSIONS: raise ValueError("ERROR! Only {} versions are supported. Not {}.".format(",".join(SUPPORTED_VERSIONS), self.version))
        self.text_segment_values["$LAST_MODIFIED"] = str(datetime.datetime.now())
        self.text_segment_values["$LAST_MODIFIER"] = "pyInfinityFlow"
        ### Prepare TEXT segment ###
        tmp_list_text_value_segments = np.array(list(self.text_segment_values.keys()))
        tmp_list_text_value_segments = np.setdiff1d(tmp_list_text_value_segments, REQUIRED_FCS_3_1)
        # Separate parameter values
        param_regex = re.compile(r'\$[a-zA-Z]+([0-9]+)[a-zA-Z]*')
        check_for_param = [param_regex.findall(item) for item in tmp_list_text_value_segments]
        mask_for_param = [item != [] for item in check_for_param]
        param_keys = [item for item, tmp_check in zip(tmp_list_text_value_segments, mask_for_param) if tmp_check]
        param_n_values = [int(item[0]) for item, tmp_check in zip(check_for_param, mask_for_param) if tmp_check]
        param_n_series = pd.Series(param_n_values, index=param_keys)
        ordered_param_keys = []
        for i in sorted(param_n_series.unique()):
            ordered_param_keys += list(sorted(param_n_series[param_n_series == i].index.values))

        tmp_list_text_value_segments = np.setdiff1d(tmp_list_text_value_segments, ordered_param_keys)
        # Ordered keys
        required_ordered_keys = list(REQUIRED_FCS_3_1) + list(ordered_param_keys)
        text_segment_list_required = []
        for tmp_key in required_ordered_keys:
            if self._check_text_segment_key(tmp_key):
                if len(self.text_segment_values[tmp_key]) > 0:
                    text_segment_list_required.append(tmp_key.replace(self.delimiter, 2 * self.delimiter))
                    text_segment_list_required.append(self.text_segment_values[tmp_key].replace(self.delimiter, 2 * self.delimiter))
                else:
                    print("WARNING! TEXT segment value for key {} is empty. Excluding from written file.".format(tmp_key))
            else:
                print("WARNING! TEXT segment value {} is not valid. Excluding from written file.".format(tmp_key))

        text_segment_str_required = (self.delimiter + self.delimiter.join(text_segment_list_required) + self.delimiter).encode('utf-8')
        text_segment_size_estimate = len(text_segment_str_required)
        if text_segment_size_estimate > FCS_TEXT_SEGMENT_SIZE_LIMIT: raise ValueError("ERROR! Required FCS properties exceed allowed size.")
        optional_ordered_keys = list(sorted(tmp_list_text_value_segments))
        text_segment_list_optional = []
        use_sup_text_segment = False
        if len(optional_ordered_keys) > 0:
            for tmp_key in optional_ordered_keys:
                if self._check_text_segment_key(tmp_key):
                    if len(self.text_segment_values[tmp_key]) > 0:
                        text_segment_list_optional.append(tmp_key.replace(self.delimiter, 2 * self.delimiter))
                        text_segment_list_optional.append(self.text_segment_values[tmp_key].replace(self.delimiter, 2 * self.delimiter))
                    else:
                        print("WARNING! TEXT segment value for key {} is empty. Excluding from written file.".format(tmp_key))
                else:
                    print("WARNING! TEXT segment value {} is not valid. Excluding from written file.".format(tmp_key))

            text_segment_str_optional = (self.delimiter.join(text_segment_list_optional) + self.delimiter).encode('utf-8')
            optional_text_segment_size_estimate = len(text_segment_str_optional)
            if (text_segment_size_estimate + optional_text_segment_size_estimate) > FCS_TEXT_SEGMENT_SIZE_LIMIT:
                use_sup_text_segment = True
                text_segment_str_optional = self.delimiter.encode('utf-8') + text_segment_str_optional
                optional_text_segment_size_estimate = len(text_segment_str_optional)
            else:
                text_segment_str_required = text_segment_str_required + text_segment_str_optional
                text_segment_size_estimate = len(text_segment_str_required)

        ### Calculate size values for DATA segment ###
        tmp_byte_length_factor = 4
        if self.text_segment_values["$DATATYPE"] == "F":
            tmp_byte_length_factor = 4
        elif self.text_segment_values["$DATATYPE"] == "D":
            tmp_byte_length_factor = 8
        else:
            raise ValueError("ERROR! Couldn't write fcs file. Only DATATYPE 'F' and 'D' are supported.")

        tmp_length_data_segment = tmp_byte_length_factor * int(self.text_segment_values["$PAR"]) * int(self.text_segment_values["$TOT"])
        ### Check if everything can be contained in the first 999,99,999 bytes of the dataset ###
        dataset_size_estimate = text_segment_size_estimate + tmp_length_data_segment
        if (dataset_size_estimate > FCS_TEXT_SEGMENT_SIZE_LIMIT) or use_sup_text_segment:
            tmp_text_segment_start = 58
            tmp_text_segment_end = 58 + text_segment_size_estimate + 120
            tmp_sup_text_segment_start = 0
            tmp_sup_text_segment_end = 0
            if use_sup_text_segment:
                tmp_sup_text_segment_start = tmp_text_segment_end + 1
                tmp_sup_text_segment_end = tmp_sup_text_segment_start + optional_text_segment_size_estimate
                tmp_data_segment_start = tmp_sup_text_segment_end + 1
                tmp_data_segment_end = tmp_data_segment_start + tmp_length_data_segment
            else:
                tmp_data_segment_start = tmp_text_segment_end + 1
                tmp_data_segment_end = tmp_data_segment_start + tmp_length_data_segment

            self.text_segment_values["$BEGINDATA"] = str(tmp_data_segment_start)
            self.text_segment_values["$BEGINSTEXT"] = str(tmp_sup_text_segment_start)
            self.text_segment_values["$ENDDATA"] = str(tmp_data_segment_end)
            self.text_segment_values["$ENDSTEXT"] = str(tmp_sup_text_segment_end)
            # Regenerate required text segment
            text_segment_list_required = []
            for tmp_key in required_ordered_keys:
                if self._check_text_segment_key(tmp_key):
                    if len(self.text_segment_values[tmp_key]) > 0:
                        text_segment_list_required.append(tmp_key.replace(self.delimiter, 2 * self.delimiter))
                        text_segment_list_required.append(self.text_segment_values[tmp_key].replace(self.delimiter, 2 * self.delimiter))
                    else:
                        print("WARNING! TEXT segment value for key {} is empty. Excluding from written file.".format(tmp_key))
                else:
                    print("WARNING! TEXT segment value {} is not valid. Excluding from written file.".format(tmp_key))

            text_segment_str_required = (self.delimiter + self.delimiter.join(text_segment_list_required) + self.delimiter).encode('utf-8')
            tmp_text_segment_end = 58 + len(text_segment_str_required)
            tmp_header_byte_str = "{}{}{}{}{}{}{}{}".format(self.version,
                                                            4*chr(32),
                                                            left_fill_int(tmp_text_segment_start),
                                                            left_fill_int(tmp_text_segment_end),
                                                            7*chr(32) + "0",
                                                            7*chr(32) + "0",
                                                            7*chr(32) + "0",
                                                            7*chr(32) + "0").encode('utf-8')

            try:
                with open(fcs_file_path, "wb") as output_fcs_file:
                    # HEADER
                    output_fcs_file.seek(0)
                    output_fcs_file.write(tmp_header_byte_str)
                    # TEXT
                    output_fcs_file.seek(tmp_text_segment_start)
                    output_fcs_file.write(text_segment_str_required)
                    if use_sup_text_segment:
                        # SUPPLEMENTAL TEXT
                        output_fcs_file.seek(tmp_sup_text_segment_start)
                        output_fcs_file.write(text_segment_str_optional)

                    # DATA SEGMENT
                    if self.text_segment_values["$DATATYPE"] == "F":
                        output_fcs_file.seek(tmp_data_segment_start)
                        output_fcs_file.write(self.data.values.flatten().astype(np.single).tostring())
                    elif self.text_segment_values["$DATATYPE"] == "D":
                        output_fcs_file.seek(tmp_data_segment_start)
                        output_fcs_file.write(self.data.values.flatten().astype(np.double).tostring())
                    else:
                        raise ValueError("ERROR! Couldn't write fcs file. Only DATATYPE 'F' and 'D' are supported.")

            except:
                raise ValueError("ERROR! Failed to write fcs file (large dataset format).")

        else:
            tmp_text_segment_start = 58
            tmp_text_segment_end = 58 + text_segment_size_estimate
            tmp_data_segment_start = tmp_text_segment_end + 1
            tmp_data_segment_end = tmp_data_segment_start + tmp_length_data_segment
            tmp_header_byte_str = "{}{}{}{}{}{}{}{}".format(self.version,
                                                            4*chr(32),
                                                            left_fill_int(tmp_text_segment_start),
                                                            left_fill_int(tmp_text_segment_end),
                                                            left_fill_int(tmp_data_segment_start),
                                                            left_fill_int(tmp_data_segment_end),
                                                            7*chr(32) + "0",
                                                            7*chr(32) + "0").encode('utf-8')
            try:
                with open(fcs_file_path, "wb") as output_fcs_file:
                    # HEADER
                    output_fcs_file.seek(0)
                    output_fcs_file.write(tmp_header_byte_str)
                    # TEXT
                    output_fcs_file.seek(tmp_text_segment_start)
                    output_fcs_file.write(text_segment_str_required)
                    # DATA SEGMENT
                    if self.text_segment_values["$DATATYPE"] == "F":
                        output_fcs_file.seek(tmp_data_segment_start)
                        output_fcs_file.write(self.data.values.flatten().astype(np.single).tostring())
                    elif self.text_segment_values["$DATATYPE"] == "D":
                        output_fcs_file.seek(tmp_data_segment_start)
                        output_fcs_file.write(self.data.values.flatten().astype(np.double).tostring())
                    else:
                        raise ValueError("ERROR! Couldn't write fcs file. Only DATATYPE 'F' and 'D' are supported.")

            except:
                raise ValueError("ERROR! Failed to write fcs file (large dataset format).")

    def to_fcs(self, fcs_file_path, add_spillover_matrix=False):
        """Write FCSFileObject to an FCS file

        Parameters
        ----------
        fcs_file_path : str
            The path to where the FCS file should be written. (Required)
        add_spillover_matrix : bool
            Specifies whether or not to add the Spillover matrix defined in the \
            FCSFileObject.spillover attribute to the TEXT segment of the FCS \
            file. (Default=False)

        Returns
        -------
        None
            Attempts to write the FCS file.

        """
        self._prepare_required_fcs_properties()
        if add_spillover_matrix: self.add_spillover_matrix()
        self._write_fcs(fcs_file_path = fcs_file_path)

    def read_fcs(self): 
        """Read data from an FCS file into the FCSFileObject instance

        This is useful if a shallow read of the FCS file was initially performed \
        without reading in the DATA segment. This function will read in the data \
        from the FCS file.

        Returns
        -------
        None
            Adds the data from the DATA segment of the fcs_file to the \
            FCSFileObject

        """
        self._read_data()




def list_fcs_channels(fcs_file_path, add_user_defined_names=False):
    """ List the channel names defined in a given FCS file

    This is useful for getting the channel names, exactly as they are written, 
    in the FCS file. It is important when defining the Reference and Query 
    channels in the backbone_annotation file, or the Target channel name in the 
    InfinityMarker annotation file that they match the names from the channels 
    in the specified FCS files.

    Note
    -------
    - "PnS" is the key used in an FCS file to specify how the user wanted to \
    annotate the channel.
    - "PnN" is the key used in an FCS file to specify the name for the channel \
    and each must be unique in a given FCS file

    Arguments
    ---------
    fcs_file_path : str
        Path to the .fcs file (Required)
    
    add_user_defined_names : bool
        If True, the function will put the user-defined ("PnS") names as a second \
        column with the unique channel ("PnN") names as the first column. \
        (Default=False)

    Returns
    -------
    None
        Prints out the channel names as a table to stdout.

    """
    # Read the header of the .fcs file
    tmp_fcs = FCSFileObject(fcs_file_path=fcs_file_path, 
        mode='r', 
        read_data_segment=False)
    # Get the channel names and print them out
    tmp_channel_anno = pd.DataFrame([tmp_fcs.named_par_channel, 
        tmp_fcs.named_par]).T
    if add_user_defined_names:
        tmp_list = (tmp_channel_anno.iloc[:,0] + ":" + tmp_channel_anno.iloc[:,1]).values
    else:
        tmp_list = tmp_channel_anno.iloc[:,0].values

    print("\n".join(tmp_list)) 


def read_binary_position(b_file, start_byte, end_byte, output_type='str', decode_type='utf-8'):
    b_file.seek(start_byte)
    if output_type == None:
        return(b_file.read(end_byte-start_byte+1))
    output_str = b_file.read(end_byte-start_byte+1).decode(decode_type)
    if output_type == 'str':
        return(output_str)
    elif output_type == 'int':
        output_str = output_str.replace(' ', '')
        if output_str == '':
            return(None)
        else:
            return(int(output_str))
    else:
        raise TypeError('\'{}\' type not defined by \'read_binary_position()\''.format(output_type))


def get_segment_from_fcs_primary_text(input_primary_text_string, segment_name):
    return(input_primary_text_string.replace("|", '\x0c').split(segment_name)[1].split('\x0c')[1])


def left_fill_int(input_int, n=8):
    return((n-len(str(input_int)))*chr(32) + str(input_int))

