#!/usr/local/bin/python3

import json
import mne
import numpy as np
import os
import pandas as pd
import warnings
from brainlife_apps_helper import helper


def notch_filter(raw, param_freqs_specific_or_start, param_freqs_end, param_freqs_step, 
                 param_picks_by_channel_types_or_names, param_picks_by_channel_indices,
                 param_filter_length, param_notch_widths, param_trans_bandwidth, param_n_jobs,
                 param_method, param_iir_parameters, param_mt_bandwidth, param_p_value,
                 param_phase, param_fir_window, param_fir_design, param_pad):
    """Apply a notch filter using MNE Python and save the file once filtered.

    Parameters
    ----------
    raw: instance of mne.io.Raw
        Data to be filtered.
    param_freqs_specific_or_start: float or None
        Specific frequency to filter out in Hz or the start of the frequencies to filter out in Hz.
    param_freqs_end: float or None
        End of the interval of frequencies to filter out in Hz. This value is excluded. 
    param_freqs_step: float or None
        The step in Hz to filter out specific frequencies (for instance the power lines harmonics) 
        between param_freqs_start and param_freqs_end.
    param_picks_by_channel_types_or_names: str, list of str, or None 
        Channels to include. In lists, channel type strings (e.g., ["meg", "eeg"]) will pick channels of those types, channel name 
        strings (e.g., ["MEG0111", "MEG2623"]) will pick the given channels. Can also be the string values “all” 
        to pick all channels, or “data” to pick data channels. None (default) will pick all data channels. Note 
        that channels in info['bads'] will be included if their names are explicitly provided.
    param_picks_by_channel_indices: list of int, slice, or None
        Channels to include. Slices (e.g., "0, 10, 2" or "0, 10" if you don't want a step) and lists of integers 
        are interpreted as channel indices. None (default) will pick all data channels. This parameter must be set 
        to None if param_picks_by_channel_types_or_names is not None. Note that channels in info['bads'] will 
        be included if their indices are explicitly provided.
    param_filter_length: str or int
        Length of the FIR filter to use (if applicable). Can be ‘auto’ (default) : the filter length is chosen based 
        on the size of the transition regions, or an other str (human-readable time in units of “s” or “ms”: 
        e.g., “10s” or “5500ms”. If int, specified length in samples. For fir_design=”firwin”, this should not be used.
    param_notch_widths: float, array of floats, or None
        Width of the stop band in Hz. If None, freqs / 200 is used. Default is None.
    param_trans_bandwidth: float
        Width of the transition band in Hz. Default is 1.
    param_n_jobs: int or str
        Number of jobs to run in parallel. Can be ‘cuda’ if cupy is installed properly and method=’fir’. Default is 1.
    param_method: str
        ‘fir’ (default) will use overlap-add FIR filtering, ‘iir’ will use IIR forward-backward filtering (via filtfilt).
        Can be ‘spectrum_fit’. 
    param_iir_parameters: dict or None
        Dictionary of parameters to use for IIR filtering. If iir_params is None and method=”iir”, 
        4th order Butterworth will be used. Default is None.
    param_mt_bandwidth: float or None
        The bandwidth of the multitaper windowing function in Hz. Default is None.
    param_p_value: float
        P-value to use in F-test thresholding to determine significant sinusoidal components 
        to remove when method=’spectrum_fit’ and freqs=None. Default is 0.05.
    param_phase: str
        Phase of the filter, only used if method='fir'. Either 'zero' (default) or 'zero-double'.
    param_fir_window: str
        The window to use in FIR design, can be “hamming” (default), “hann”, or “blackman”.
    param_fir_design: str
        Can be “firwin” (default) or “firwin2”.
    param_pad: str
        The type of padding to use. Supports all numpy.pad() mode options. Can also be “reflect_limited” (default).

    Returns
    -------
    raw_filtered: instance of mne.io.Raw
        The raw data after filtering.
    """

    raw.load_data()

    # Raise error if both param picks are not None
    if param_picks_by_channel_types_or_names is not None and param_picks_by_channel_indices is not None:
        value_error_message = f"You can't provide values for both " \
                              f"param_picks_by_channel_types_or_names and " \
                              f"param_picks_by_channel_indices. One of them must be " \
                              f"set to None."
        raise ValueError(value_error_message)
    # Define param_picks
    elif param_picks_by_channel_types_or_names is None and param_picks_by_channel_indices is not None:
        param_picks = param_picks_by_channel_indices
    elif param_picks_by_channel_types_or_names is not None and param_picks_by_channel_indices is None:
        param_picks = param_picks_by_channel_types_or_names
    else:
        param_picks = None       

    # Notch between two frequencies
    if param_freqs_end is not None:
        freqs = np.arange(param_freqs_specific_or_start, param_freqs_end, param_freqs_step)
    # Notch one frequency
    else:
        freqs = param_freqs_specific_or_start

    raw_notch_filtered = raw.notch_filter(freqs=freqs, picks=param_picks,
                                          filter_length=param_filter_length, notch_widths=param_notch_widths,
                                          trans_bandwidth=param_trans_bandwidth, n_jobs=param_n_jobs,
                                          method=param_method, iir_params=param_iir_parameters,
                                          mt_bandwidth=param_mt_bandwidth, p_value=param_p_value,
                                          phase=param_phase, fir_window=param_fir_window,
                                          fir_design=param_fir_design, pad=param_pad)

    # Save file
    raw_notch_filtered.save("out_dir_notch_filter/meg.fif", overwrite=True)

    return raw_notch_filtered


def _compute_snr(meg_file):
    # Compute the SNR

    # select only MEG channels and exclude the bad channels
    meg_file = meg_file.pick_types(meg=True, exclude='bads')

    # create fixed length events
    array_events = mne.make_fixed_length_events(meg_file, duration=10)

    # create epochs
    epochs = mne.Epochs(meg_file, array_events)

    # mean signal amplitude on each epoch
    epochs_data = epochs.get_data()
    mean_signal_amplitude_per_epoch = epochs_data.mean(axis=(1, 2))  # mean on channels and times

    # mean across all epochs and its std error
    mean_final = mean_signal_amplitude_per_epoch.mean()
    std_error_final = np.std(mean_signal_amplitude_per_epoch, ddof=1) / np.sqrt(
        np.size(mean_signal_amplitude_per_epoch))

    # compute SNR
    snr = mean_final / std_error_final

    return snr


def _generate_report(data_file_before, raw_before_preprocessing, raw_after_preprocessing, bad_channels,
                     comments_notch, param_freqs_specific_or_start, param_freqs_end, param_freqs_step, 
                     param_picks_by_channel_types_or_names, param_picks_by_channel_indices,
                     param_filter_length, param_notch_widths, param_trans_bandwidth, param_n_jobs,
                     param_method, param_iir_parameters, param_mt_bandwidth, param_p_value,
                     param_phase, param_fir_window, param_fir_design, param_pad):
    # Generate a report

    # Instance of mne.Report
    report = mne.Report(title='Results of filtering ', verbose=True)

    ## Give some info about the file before preprocessing ##

    # Check if MaxFilter was already applied on the data # 
    if raw_before_preprocessing.info['proc_history']:
        sss_info = raw_before_preprocessing.info['proc_history'][0]['max_info']['sss_info']
        tsss_info = raw_before_preprocessing.info['proc_history'][0]['max_info']['max_st']
        if bool(sss_info) or bool(tsss_info) is True:
            message_channels = f'Bad channels have been interpolated during MaxFilter'
        else:
            message_channels = bad_channels
    else:
        message_channels = bad_channels

    # Give some info about the file before preprocessing
    sampling_frequency = raw_before_preprocessing.info['sfreq']
    highpass = raw_before_preprocessing.info['highpass']
    lowpass = raw_before_preprocessing.info['lowpass']

    # Put this info in html format # 
    html_text_info = f"""<html>

        <head>
            <style type="text/css">
                table {{ border-collapse: collapse;}}
                td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
            </style>
        </head>

        <body>
            <table width="50%" height="80%" border="2px">
                <tr>
                    <td>Input file: {data_file_before}</td>
                </tr>
                <tr>
                    <td>Bad channels: {message_channels}</td>
                </tr>
                <tr>
                    <td>Sampling frequency before preprocessing: {sampling_frequency}Hz</td>
                </tr>
                <tr>
                    <td>Highpass before preprocessing: {highpass}Hz</td>
                </tr>
                <tr>
                    <td>Lowpass before preprocessing: {lowpass}Hz</td>
                </tr>
            </table>
        </body>

        </html>"""

    # Add html to reports
    report.add_htmls_to_section(html_text_info, captions='Data recording features', section='Data info', replace=False)

    # Define param_picks
    if param_picks_by_channel_types_or_names is None and param_picks_by_channel_indices is not None:
        param_picks = param_picks_by_channel_indices
    elif param_picks_by_channel_types_or_names is not None and param_picks_by_channel_indices is None:
        param_picks = param_picks_by_channel_types_or_names
    else:
        param_picks = None       

    fig_raw = raw_before_preprocessing.pick(param_picks, exclude='bads').plot(duration=10, scalings='auto', butterfly=False,
                                                                              show_scrollbars=False, proj=False, show=False)
    fig_raw_maxfilter = raw_after_preprocessing.pick(param_picks, exclude='bads').plot(duration=10, scalings='auto',
                                                                                       butterfly=False,
                                                                                       show_scrollbars=False, 
                                                                                       proj=False, show=False)

    # Plot power spectral density
    fig_raw_psd = raw_before_preprocessing.plot_psd(picks=param_picks, show=False)
    fig_raw_maxfilter_psd = raw_after_preprocessing.plot_psd(picks=param_picks, show=False)

    # Add figures to report
    report.add_figs_to_section(fig_raw, captions='Signals before notch filtering', section='Temporal domain')
    report.add_figs_to_section(fig_raw_maxfilter, captions='Signals after notch filtering',
                               comments=f'Notch Filter: {comments_notch}',
                               section='Temporal domain')
    report.add_figs_to_section(fig_raw_psd, captions='Power spectral density before notch filtering',
                               section='Frequency domain')
    report.add_figs_to_section(fig_raw_maxfilter_psd, captions='Power spectral density after notch filtering',
                               comments=f'Notch Filter: {comments_notch}',
                               section='Frequency domain')

    # Info on SNR
    # html_text_snr = f"""<html>

    # <head>
    #     <style type="text/css">
    #         table {{ border-collapse: collapse;}}
    #         td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
    #     </style>
    # </head>

    # <body>
    #     <table width="50%" height="80%" border="2px">
    #         <tr>
    #             <td>SNR before filtering: {snr_before}</td>
    #         </tr>
    #         <tr>
    #             <td>SNR after filtering: {snr_after}</td>
    #         </tr>
    #     </table>
    # </body>

    # </html>"""

    # report.add_htmls_to_section(html_text_snr, captions='Signal to noise ratio', section='Signal to noise ratio',
    #                             replace=False)


    ## Values of the parameters of the App ## 
    mne_version = mne.__version__

    # Put this info in html format # 
    html_text_parameters = f"""<html>

    <head>
        <style type="text/css">
            table {{ border-collapse: collapse;}}
            td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
        </style>
    </head>

    <body>
        <table width="50%" height="80%" border="2px">
            <tr>
                <td>Frequency to notch filter: {param_freqs_specific_or_start} Hz</td>
            </tr>
            <tr>
                <td>End of the interval of frequencies to filter out: {param_freqs_end} Hz</td>
            </tr>
            <tr>
                <td>Step to filter out specific frequency between the two previous frequencies: {param_freqs_step} Hz</td>
            </tr>
            <tr>
                <td>Types or names of channels to include: {param_picks_by_channel_types_or_names}</td>
            </tr>
            <tr>
                <td>Indices of channels to include: {param_picks_by_channel_indices}</td>
            </tr>
            <tr>
                <td>Length of the FIR filter {param_filter_length}</td>
            </tr>
            <tr>
                <td>Widths of the stop band: {param_notch_widths} Hz</td>
            </tr>
            <tr>
                <td>Width of the transition band: {param_trans_bandwidth}</td>
            </tr>
            <tr>
                <td>Number of jobs to run in parallel: {param_n_jobs}</td>
            </tr>
            <tr>
                <td>Filtering method: {param_method}</td>
            </tr>
            <tr>
                <td>IIR parameters: {param_iir_parameters}</td>
            </tr>
            <tr>
                <td>Bandwidth of the multitaper windowing: {param_mt_bandwidth} Hz</td>
            </tr>
            <tr>
                <td>p-value: {param_p_value}</td>
            </tr>
            <tr>
                <td>Phase of the filter: {param_phase}</td>
            </tr>
            <tr>
                <td>FIR window: {param_fir_window}</td>
            </tr>
            <tr>
                <td>FIR design: {param_fir_design}</td>
            </tr>
            <tr>
                <td>Type of padding: {param_pad}</td>
            </tr>
            <tr>
                <td>MNE version used: {mne_version}</td>
            </tr>
        </table>
    </body>

    </html>"""

    # Add html to reports
    report.add_htmls_to_section(html_text_parameters, captions='Values of the parameters of the App', 
                                section='Parameters of the App', replace=False)

    # Save report
    report.save('out_dir_report/report_notch_filtering.html', overwrite=True)


def main():

    # Generate a json.product to display messages on Brainlife UI
    dict_json_product = {'brainlife': []}

    # Load inputs from config.json
    with open('config.json') as config_json:
        config = json.load(config_json)

    # Read the files
    data_file = config.pop('fif')
    raw = mne.io.read_raw_fif(data_file, allow_maxshield=True)

    # Read and save optional files
    config, cross_talk_file, calibration_file, events_file, head_pos_file, channels_file, destination = helper.read_optional_files(config, 'out_dir_notch_filter')
    
    # Convert empty strings values to None
    config = helper.convert_parameters_to_None(config)

    # Raise error if no start parameter and method is wrong 
    if config['param_freqs_specific_or_start'] is None and config['param_method'] != 'spectrum_fit':
        value_error_message = f'This frequency can only be None when method is spectrum_fit.' 
        # Raise exception
        raise ValueError(value_error_message)

    
    ## Convert parameters ## 

    # Deal with param_picks_by_channel_indices parameter #
    # Convert it into a slice When the App is run locally and on BL
    picks = config['param_picks_by_channel_indices']
    if isinstance(picks, str) and picks.find(",") != -1 and picks.find("[") == -1 and picks is not None:
        picks = list(map(int, picks.split(', ')))
        if len(picks) == 2:
            config['param_picks_by_channel_indices'] = slice(picks[0], picks[1])
        elif len(picks) == 3:
            config['param_picks_by_channel_indices'] = slice(picks[0], picks[1], picks[2])
        else:
            value_error_message = f"If you want to select channels using a slice, you must give two or three elements."
            raise ValueError(value_error_message)

    # Convert it into a list of int when the app is run on BL
    if isinstance(picks, str) and picks.find(",") != -1 and picks.find("[") != -1 and picks is not None:
        picks = picks.replace('[', '')
        picks = picks.replace(']', '')
        config['param_picks_by_channel_indices'] = list(map(int, picks.split(', ')))

    # Deal with param_picks_by_channel_types_or_name parameter #
    # Convert it into a list of str when the App is run on BL
    picks = config['param_picks_by_channel_types_or_names']
    if isinstance(picks, str) and picks.find("[") != -1 and picks is not None:
        picks = picks.replace('[', '')
        picks = picks.replace(']', '')
        config['param_picks_by_channel_types_or_names'] = list(map(str, picks.split(', ')))

    # Deal with filter_length parameter #
    # Convert it into int if not "auto" when the App runs on BL
    if config['param_filter_length'] != "auto" and config['param_filter_length'].find("s") == -1:
        config['param_filter_length'] = int(config['param_filter_length'])

    # Deal with param_notch_widths parameter #
    # Convert notch widths parameter into array when the app is run locally
    if isinstance(config['param_notch_widths'], list):
       config['param_notch_widths'] = np.array(config['param_notch_widths'])

    # Convert notch widths parameter into array when the app is run on BL
    if isinstance(config['param_notch_widths'], str):
        config['param_notch_widths'] = list(map(float, config['param_notch_widths'].split(', ')))
        if len(config['param_notch_widths']) == 1:
            config['param_notch_widths'] = float(config['param_notch_widths'][0])
        else:
            config['param_notch_widths'] = np.array(config['param_notch_widths'])

    # Deal with param_n_jobs parameter #
    # Convert it into into if not "cuda" When the App is run on BL
    if config['param_n_jobs'] != 'cuda':
        config['param_n_jobs']  = int(config['param_n_jobs'])

    
    # Comments messages about filtering #
    if config['param_freqs_specific_or_start'] is not None and config['param_freqs_end'] is None:
        comments_notch = f"{config['param_freqs_specific_or_start']}Hz" 
    elif config['param_freqs_specific_or_start'] is not None and config['param_freqs_end'] is not None:  
        comments_notch = f"Between {config['param_freqs_specific_or_start']} and {config['param_freqs_end']}Hz"
        if config['param_freqs_step'] is not None:  
            comments_notch = f"Between {config['param_freqs_specific_or_start']} and " \
                             f"{config['param_freqs_end']}Hz every {config['param_freqs_step']}Hz"

    # Channels.tsv must be BIDS compliant
    if channels_file is not None:
        user_warning_message_channels = f'The channels file provided must be ' \
                                        f'BIDS compliant and the column "status" must be present. ' 
        warnings.warn(user_warning_message_channels)
        dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message_channels})
        # Udpate raw.info['bads'] with info contained in channels.tsv
        raw, user_warning_message_channels = helper.update_data_info_bads(raw, channels_file)
        if user_warning_message_channels is not None: 
            warnings.warn(user_warning_message_channels)
            dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message_channels})

    # Keep bad channels in memory #
    bad_channels = raw.info['bads']

    # Delete keys values in config.json when this app is executed on Brainlife
    kwargs = helper.define_kwargs(config) 

    # Apply temporal filtering
    raw_copy = raw.copy()
    raw_notch_filtered = notch_filter(raw_copy, **kwargs)
    del raw_copy

    # Success message in product.json    
    dict_json_product['brainlife'].append({'type': 'success', 'msg': 'Notch filter was applied successfully.'})

    # Compute SNR
    # snr_before = _compute_snr(raw)
    # snr_after = _compute_snr(raw_filtered)

    # Generate a report
    _generate_report(data_file, raw, raw_notch_filtered, bad_channels,
                     comments_notch, **kwargs)

    # Save the dict_json_product in a json file
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)
        
    # == Save Ephys derivatives json file ==
    # Create new .json file, with the following fields: 
    #  "Description": "Data filtering"
    #  "RawSources": ["xxxx"] // paths relative to dataset root pointing to the BIDS-Raw file(s) that were used in the creation of this derivative
    #  "Sources": ["xxxx"] // paths to other derivatives used
    #  "Filters": {
    #        "<Filter1Name>": {"<Filter1Property1>": "<Filter1Setting1>"},
    #         },... 




if __name__ == '__main__':
    main()
