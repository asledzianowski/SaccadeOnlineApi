import os
import traceback
import pandas as pd
import saccade_finder
import numpy as np
import json

def analyze():
    try:

        id = 135447
        #desktop #101633 #181718 #190914 #191038
        file_path = r'../results/online/out_' + str(id) + '.csv'
        time_string = id
        experiment_data_df = pd.read_csv(r'../results/online/out_' + str(id) + '.csv')
        calibration_data_df = pd.read_csv(r'../results/online/calibration_out_' + str(id) + '.csv')
        frq = get_frequency_for_segment(experiment_data_df, 3000, 4000)
        print(frq)
        exit()

        calibration_means = get_calibration_means(calibration_data_df)
        experiment_data_df.loc[experiment_data_df['marker'] == 0, 'marker'] = calibration_means['mean_0']
        experiment_data_df.loc[experiment_data_df['marker'] == 1, 'marker'] = calibration_means['mean_1']
        experiment_data_df.loc[experiment_data_df['marker'] == -1, 'marker'] = calibration_means['mean_minus_1']


        df_parameters, graph_plt = saccade_finder.analyze_result(experiment_data_df, time_string)
        json_list = json.loads(json.dumps(list(df_parameters.T.to_dict().values())))
        print(json_list)
        graph_plt.show()

    except Exception as ex:
        print(ex)
        print(traceback.print_exc())

def get_calibration_means(calibration_data):

    # It was confirmed that, for sampling frequency lower than 25Hz, the display time of a calibration
    # point can be as short as 1250 ms after a moment when an eye signal becomes stable.
    # https://www.sciencedirect.com/science/article/pii/S1877050914011594
    # remove first 2000 ms
    eye_stabilization_peroid = 2000
    calibration_data_0 = calibration_data[calibration_data["state"] == 0]
    calibration_data_1 = calibration_data[calibration_data["state"] == 1]
    calibration_data_minus_1 = calibration_data[calibration_data["state"] == -1]

    start_time_data_0 = calibration_data_0['time'].iloc[0] + eye_stabilization_peroid
    start_time_data_1 = calibration_data_1['time'].iloc[0] + eye_stabilization_peroid
    start_time_data_minus_1 = calibration_data_minus_1['time'].iloc[0] + eye_stabilization_peroid

    filetered_data_0 = calibration_data_0[calibration_data_0['time'] > start_time_data_0]
    filetered_data_1 = calibration_data_1[calibration_data_1['time'] > start_time_data_1]
    filetered_data_minus_1 = calibration_data_minus_1[calibration_data_minus_1['time'] > start_time_data_minus_1]

    means = {
        'mean_0': np.round(np.mean(filetered_data_0["gaze_x"]), 2),
        'mean_1': np.round(np.mean(filetered_data_1["gaze_x"]), 2),
        'mean_minus_1': np.round(np.mean(filetered_data_minus_1["gaze_x"]), 2)}
    return means


def get_frequency_for_segment(df, start_index, end_index):
    raw_time = df['time'].to_numpy()
    over_one = np.where(raw_time >= start_index)
    index_over_two = np.where(raw_time[over_one[0][0]:] >= end_index)
    tracker_frequency = index_over_two[0][0]
    return tracker_frequency

if __name__ == "__main__":
    analyze()