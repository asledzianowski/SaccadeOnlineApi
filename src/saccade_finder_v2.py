# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import warnings
import dataclasses
import math

warnings.filterwarnings('ignore')
# %matplotlib inline
plt.rcParams["figure.figsize"] = (20, 8)


def atan(x):
    return np.arctan(x)


class SaccadeFinderV2:

    def __init__(
            self,
            # freq_hz,
            # distance_from_screen,
            # screen_resolution_pixels,
            # screen_width_mm
            freq_hz = 14,
            distance_from_screen = 86,
            screen_resolution_pixels = (1920, 1080),
            screen_width_mm = 544
    ):

        self.freq_hz = freq_hz
        self.distance_from_screen = distance_from_screen
        self.screen_resolution = screen_resolution_pixels
        self.screen_width = screen_width_mm

        # self.min_saccade_latency_ms = 40
        # self.max_saccade_latency_ms = 600
        #
        # self.min_saccade_duration_ms = 60
        # self.max_saccade_duration_ms = 350

        self.min_saccade_latency_ms = 40
        self.max_saccade_latency_ms = 600
        self.min_saccade_duration_ms = 10
        self.max_saccade_duration_ms = 350

        # saccade amplitude should be greater than standard deviation * min_saccade_std_factor
        self.min_saccade_std_factor = 1.5
        # ---------

        self.period_ms = 1 / freq_hz * 1000
        self.period_ms_round = round(self.period_ms)
        # 1 because difference between 2 indexes is calculated
        self.minimum_saccade_indexes = math.ceil(self.min_saccade_duration_ms / self.period_ms_round) + 1

    def read_df(self, folder_type, file_id):
        filepath = os.path.join(folder_type, f'out_{file_id}.csv')
        df = pd.read_csv(filepath)
        df = df.set_index('time')
        return df

    def find_signal_changes(self, df, col):
        '''
        1 - saccade up
        -1 - saccade down
        '''
        diff = df[col].diff()
        return diff[diff != 0].iloc[1:]

    @dataclasses.dataclass
    class SaccadeParameters:
        Start: float
        End: float
        MarkerChange: float
        LatencyFrameCount: int
        DurationFrameCount: int
        Latency: float
        Duration: float
        Amplitude: float = None  # degrees
        Distance: float = None  # cm on screen
        Velocity: float = None  # deg/s
        AvgVelocity: float = None  # average per each frame
        MaxVelocity: float = None  # max in all saccade frames
        Gain: float = None

    def plot_saccades(self, df_s, sac, tittle):
        ax = df_s.dropna()[['gaze_x', 'marker']].plot()
        for s, e in sac:
            ax.plot(s, df_s.loc[s, 'gaze_x'], marker="o", markersize=9, markerfacecolor="green")
            ax.plot(e, df_s.loc[e, 'gaze_x'], marker="o", markersize=9, markerfacecolor="red")

        ax.set_title(tittle)
        return plt

    def plot_saccades_and_numbers(self, df_s, sac, tittle):
        df_s = df_s.set_index('time')
        ax = df_s.dropna()[['gaze_x', 'marker']].plot(linewidth=1)
        idx = 0
        for s, e in sac:
            ax.plot(s, df_s.loc[s, 'gaze_x'], marker="o", markersize=9, markerfacecolor="green")
            ax.plot(e, df_s.loc[e, 'gaze_x'], marker="o", markersize=9, markerfacecolor="red")
            plt.text(s, df_s.loc[s, 'gaze_x'] - 0.8, str(idx), horizontalalignment='left', weight='heavy')
            idx += 1

        ax.set_title(tittle)
        return plt



    # +
    def get_signal_changes_with_direction(self, df):
        signal_changes_df = self.find_signal_changes(df, 'state')
        direction = (-signal_changes_df).tolist()
        signal_changes_index = signal_changes_df.index.tolist()
        return signal_changes_index, direction

    # def split_df_with_direction(self, df, direction, signal_changes_index):
    #     part_dfs = [df.loc[idx:idx + 5000] for idx in signal_changes_index]
    #     part_df_direction = list(zip(part_dfs, direction, signal_changes_index))
    #     return part_df_direction

    def take_closest(self, num, collection):
        return min(collection, key=lambda x: abs(x - num))

    # def split_df_with_direction(self, df, direction, signal_changes_index):
    #     part_dfs = [df.loc[idx:idx + 5000] for idx in signal_changes_index]
    #     part_df_direction = list(zip(part_dfs, direction, signal_changes_index))
    #     return part_df_direction

    def split_df_with_direction(self, df, direction, signal_changes_index):
        try:
            part_dfs = []
            for idx in signal_changes_index:
                proper_idx = self.take_closest(idx + 5000, df['time_bac'].to_numpy())
                part_df = df.loc[idx:proper_idx]
                part_dfs.append(part_df)
            #part_dfs = [df.loc[idx:idx + 5000] for idx in signal_changes_index]
            part_df_direction = list(zip(part_dfs, direction, signal_changes_index))
            return part_df_direction
        except Exception as e:
            print(e)

    # -

    def find_single_saccade(self, part_df, direction):
        '''direction = 1 signal going up
        direction -1 signal goind down'''
        try:
            part_df['diff'] = part_df.gaze_x.diff().dropna()
            start_idx = part_df.index.tolist()[0]
            std = part_df.gaze_x.std()

            if direction == 1:
                part_df['diff_binary'] = part_df['diff'].apply(lambda x: 1 if x < 0.01 else 0)
            else:
                part_df['diff_binary'] = part_df['diff'].apply(lambda x: 1 if x > -0.01 else 0)

            groups = part_df[['diff', 'diff_binary']].ne(0).cumsum()
            clusters = part_df.state.groupby(groups['diff_binary']).agg(list)

            cluster_min_len = clusters[clusters.apply(len) >= self.minimum_saccade_indexes].reset_index()

            cluster_min_len['group_idx'] = cluster_min_len.diff_binary \
                .apply(lambda x: groups[groups.diff_binary == x].index.tolist())

            cluster_min_len['change'] = cluster_min_len.group_idx.apply(
                lambda x: (part_df.loc[x[-1]] - part_df.loc[x[0]]).gaze_x
            )
            cluster_min_len['latency'] = cluster_min_len.group_idx.apply(
                lambda x: x[0] - start_idx
            )

            cluster_min_len['duration'] = cluster_min_len.group_idx.apply(
                lambda x: x[1] - x[0]
            )

            cluster_min_len['amplitude'] = cluster_min_len.change.abs()

            cluster_min_len = cluster_min_len.query(
                f'{self.min_saccade_latency_ms} < latency < {self.max_saccade_latency_ms}')
            cluster_min_len = cluster_min_len.query(
                f'{self.min_saccade_duration_ms} < duration < {self.max_saccade_duration_ms}')

            cluster_min_len = cluster_min_len.query(f'{std * self.min_saccade_std_factor} < amplitude')
            if direction == 1:
                cluster_min_len = cluster_min_len.query('change > 0')
            else:
                cluster_min_len = cluster_min_len.query('change < 0')
            cluster_idx = cluster_min_len.query(f'abs(change) > {std}').iloc[0]
            saccade_start_idx = cluster_idx.group_idx[0]
            saccade_end_idx = cluster_idx.group_idx[-1]
            #         single saccade plot
            #         plot_saccades(df, [(saccade_start_idx, saccade_end_idx)], file_id)

            return (saccade_start_idx, saccade_end_idx)
        except IndexError as e:
            print('There was an error looking for a saccade', e)
            return None

    def calculate_saccade_params_df(self, df, saccades, signal_change_idx):
        sac_parameters = []
        df = df.eval("""
            gaze_x_diff = gaze_x.diff(1).abs()
            pixel_distance = @self.screen_resolution[0] * gaze_x_diff 
            distance_cm = @self.pixels_to_cm(pixel_distance)
            tangens = distance_cm / @self.distance_from_screen
            
            amplitude_rad = gaze_x 
            amplitude = gaze_x_diff

            duration_s = @df.index.to_series().diff(1) / 1000
            velocity = amplitude / duration_s

            marker_x_diff = marker.diff(1).abs()
            marker_distance_pixel = @self.screen_resolution[0] * marker.diff(1).abs()
            marker_distance_cm = @self.pixels_to_cm(marker_distance_pixel)
            marker_tangens = marker_distance_cm / @self.distance_from_screen
            
            marker_amplitude_rad = marker
            marker_amplitude = marker_x_diff
        """
                     )

        for sac, marker in zip(saccades, signal_change_idx):
            try:
                # a = df.gaze_x[sac[0]]
                # b = df.gaze_x[sac[1]]
                # scaled_a = self.screen_resolution[0] * a
                # scaled_b = self.screen_resolution[0] * b

                # pixel_distance = np.abs(scaled_a - scaled_b)
                # distance_on_screen_cm = self.pixels_to_cm(pixel_distance)

                # tangens = distance_on_screen_cm / self.distance_from_screen
                # ampInRadians = np.arctan(tangens)
                # amplitude = self.RadianToDegree(ampInRadians)

                # velocity = amplitude / duration_s # deg/s

                duration_s = (sac[1] - sac[0]) / 1000
                latency = (sac[0] - marker)
                saccade_df = df.loc[sac[0] + 1 :sac[1]]  # +1 because of shift when using diff function
                amplitude = saccade_df.amplitude.sum()
                velocity = saccade_df.amplitude.sum() / saccade_df.duration_s.sum()
                avg_velocity = saccade_df.velocity.mean()
                max_velocity = saccade_df.velocity.max()
                distance_on_screen_cm = saccade_df.distance_cm.sum()

                last_marker_amplitude = (
                    df.loc[:sac[0], ['marker_amplitude']]
                    .query('marker_amplitude > 0')
                    .marker_amplitude.iloc[-1]
                )
                gain = amplitude / last_marker_amplitude

                sac_param = self.SaccadeParameters(
                    Start=sac[0],
                    End=sac[1],
                    MarkerChange=marker,
                    LatencyFrameCount=int((sac[0] - marker) / self.period_ms_round),
                    DurationFrameCount=int((sac[1] - sac[0]) / self.period_ms_round),
                    Latency=latency,
                    Duration=duration_s * 1000,
                    Amplitude=round(amplitude, 2),
                    Distance=distance_on_screen_cm,
                    Velocity=round(velocity, 2),
                    AvgVelocity=round(avg_velocity, 2),
                    MaxVelocity=round(max_velocity, 2),
                    Gain=round(gain, 2)
                )
                time_str = f'{sac_param.Start} - {sac_param.End}'
                assert sac_param.Latency > self.min_saccade_latency_ms, f'Saccade latency was below 30ms, ({time_str})'
                assert sac_param.Duration > 30, f'Saccade duration was below 30ms, ({time_str})'
                assert sac_param.Start > sac_param.MarkerChange, 'Saccade before marker change'
                sac_parameters.append(sac_param.__dict__)
            except AssertionError as e:
                print(e)
        df_parameters = pd.DataFrame(sac_parameters)
        # df_parameters['TotalAvgVelocity'] = df_parameters['Velocity'].mean()
        # df_parameters['TotalMaxVelocity'] = df_parameters['Velocity'].max()
        return df_parameters

    def RadianToDegree(self, angle):
        return angle * (180.0 / math.pi)

    def pixels_to_cm(self, pixel_dist):
        pixel_to_mm = self.screen_width / self.screen_resolution[0]
        distance_mm = pixel_to_mm * pixel_dist
        distance_cm = distance_mm / 10
        return distance_cm

    def find_saccades(self, df):
        all_saccades = []
        signal_changes_sac_detected = []

        # look for signal changes and get the direction 1 - up, -1 down
        signal_changes_index, direction = self.get_signal_changes_with_direction(df)
        # zip signal_change_indexes with directions and parts of the dataframe
        part_df_direction = self.split_df_with_direction(df, direction, signal_changes_index)

        for part_df, direction_change, signal_change_idx in part_df_direction:
            saccade = self.find_single_saccade(part_df, direction_change)
            if saccade:
                all_saccades.append(saccade)
                signal_changes_sac_detected.append(signal_change_idx)

        df_parameters = self.calculate_saccade_params_df(df, all_saccades, signal_changes_sac_detected)

        return df_parameters, all_saccades

    def analyze_result(self, df, id):
        df["time_bac"] = df['time']
        df = df.set_index('time')
        df_parameters, all_saccades = self.find_saccades(df)
        graph_plt = self.plot_saccades(df, all_saccades, id)
        return df_parameters, graph_plt


    def analyze_result_no_plt(self, df):
        df["time_bac"] = df['time']
        df = df.set_index('time')
        df_parameters, all_saccades = self.find_saccades(d)
        return df_parameters, all_saccades

# sf = SaccadeFinder()
# df = pd.read_csv('data/out_183318.csv')
# sf.analyze_result(df, '135714')
