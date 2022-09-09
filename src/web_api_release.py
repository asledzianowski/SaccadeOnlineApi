from gevent import monkey

monkey.patch_all()
import sys

sys.path.insert(0, './common')
sys.path.insert(0, './openvino_classifications')
sys.path.insert(0, './calculations')
sys.path.insert(0, './camera')
sys.path.insert(0, './experiment')
sys.path.insert(0, './calibration')

import bottle
from bottle import route, run, request, response

import base64
import matplotlib.pyplot as plt
import io
from datetime import datetime
from json import dumps
import base64
from PIL import Image
import numpy as np
import cv2 as cv
import pandas as pd
from io import BytesIO
from scipy.signal import periodogram
from openvino_classifications.classification_manager import ClassificationManager
from saccade_finder import SaccadeFinder
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


classifiers = None
bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024
results_output_folder_path = r'../results/online/'


def enable_cors(fn):
    def _enable_cors(*args, **kwargs):
        # set CORS domain for live server
        response.headers['Access-Control-Allow-Origin'] = 'brainonline.pja.edu.pl'
        #response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
        response.headers[
            'Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

        if bottle.request.method != 'OPTIONS':
            # actual request; reply with the actual response
            return fn(*args, **kwargs)

    return _enable_cors


@route(path='/test', method=['OPTIONS', 'GET'])
@enable_cors
def test():
    response.content_type = 'application/json'
    output_data = dumps("OK FROM API")
    return output_data


@route(path='/getfacedata', method=['OPTIONS', 'POST'])
@enable_cors
def get_face_data():
    try:
        base64_image_string = request.json['base64_image_string']
        cv_image = readb64(base64_image_string)

        face_boxes = classifiers.predict_face_position(cv_image)
        # height, width, channels = cv_image.shape
        # write_debug_image(cv_image, face_boxes)

        response.content_type = 'application/json'
        output_data = dumps({'face_boxes': face_boxes})
        return output_data

    except Exception as ex:
        print(ex)
        # response.status = 500
        return str(ex)


@route(path='/getfaceandeyedata', method=['OPTIONS', 'POST'])
@enable_cors
def get_face_and_eye_data():
    try:
        base64_image_string = request.json['base64_image_string']
        cv_image = readb64(base64_image_string)
        face_eye_boxes = []
        face_eye_centers = []
        face_boxes = classifiers.predict_face_position(cv_image)
        for current_face_box in face_boxes:
            face_image = classifiers.get_crop_image(cv_image, current_face_box)
            eye_boxes, eye_centers = classifiers.facial_landmarks_detector.predict(face_image)
            face_eye_boxes.append(eye_boxes)
            face_eye_centers.append(eye_centers)

        response.content_type = 'application/json'
        output_data = dumps(
            {'face_boxes': face_boxes,
             'face_eye_boxes': face_eye_boxes,
             'face_eye_centers': face_eye_centers
             })
        return output_data

    except Exception as ex:
        print(ex)
        # response.status = 500
        return str(ex)


@route(path='/postgazedata', method=['OPTIONS', 'POST'])
@enable_cors
def process_gaze_data():
    try:

        base64_image_string = request.json['base64_image_string']
        cv_image = readb64(base64_image_string)
        gaze_x, gaze_y = classifiers.predict_gaze_position(cv_image)
        gaze_x = np.round(gaze_x, 2)
        gaze_y = np.round(gaze_y, 2)
        response.content_type = 'application/json'
        output_data = dumps({'gaze_x': gaze_x, 'gaze_y': gaze_y})
        return output_data

    except Exception as ex:
        print(ex)
        # response.status = 500
        return str(ex)


@route(path='/postresultsdata', method=['OPTIONS', 'POST'])
@enable_cors
def process_results_data():
    try:
        # convert input data
        calibration_data = request.json['calibration_data']
        experiment_data = request.json['experiment_data']
        distance_from_screen = request.json['distance_from_screen'] #86
        screen_resolution = request.json['screen_resolution']
        screen_width_mm = request.json['screen_width_mm']
        form_data = request.json['form_data']
        
        print("distance_from_screen: ", distance_from_screen)
        print("screen_resolution: ", screen_resolution)
        print("screen_width_mm: ", screen_width_mm)

        # save raw data
        now = datetime.now()
        time_string = now.strftime("%H%M%S")

        #save form data
        if form_data is not None:
            with open(results_output_folder_path + 'form_data_' + time_string + '.json', 'w') as f:
                json.dump(form_data, f)

        calibration_data_df = pd.DataFrame(calibration_data)
        experiment_data_df = pd.DataFrame(experiment_data)
        experiment_data_df = experiment_data_df.sort_values(by=['time'])
        calibration_data_df = calibration_data_df.sort_values(by=['time'])
        # calculate and apply calibration data
        calibration_means = get_calibration_means(calibration_data_df)
        experiment_data_df.loc[experiment_data_df['marker'] == 0, 'marker'] = calibration_means['mean_0']
        experiment_data_df.loc[experiment_data_df['marker'] == 1, 'marker'] = calibration_means['mean_1']
        experiment_data_df.loc[experiment_data_df['marker'] == -1, 'marker'] = calibration_means['mean_minus_1']
        freq = get_frequency_for_segment(experiment_data_df, 1000, 2000)

        calibration_data_df.to_csv(results_output_folder_path +
                                   'calibration_out_' + time_string + '.csv', index=False)
        experiment_data_df.to_csv(results_output_folder_path + 'out_' + time_string +
                                  '_' + str(freq) + 'hz' + '.csv', index=False)
        # analyze data
        saccade_finder = SaccadeFinder(freq, distance_from_screen, screen_resolution, screen_width_mm)
        df_parameters, graph_plt = saccade_finder.analyze_result(experiment_data_df, time_string)
        # save results
        df_parameters.to_csv(results_output_folder_path + 'calculations_out_' + time_string + '.csv', index=False)
        # save graph
        graph_plt.savefig(results_output_folder_path + 'chart_out_' + time_string + '.png')

        # convert results
        json_list = json.loads(json.dumps(list(df_parameters.T.to_dict().values())))
        # convert plot
        pic_bytes = io.BytesIO()
        graph_plt.savefig(pic_bytes, format='png')
        pic_bytes.seek(0)
        base64_bytes = base64.b64encode(pic_bytes.read())
        base64_string = base64_bytes.decode('utf-8')
        graph_plt.close('all')

        # return result
        response.content_type = 'application/json'
        output_data = dumps({
            'result_id': time_string,
            'result_freq': str(freq),
            'result_image': base64_string,
            'result_data': json_list
        })
        return output_data

    except Exception as ex:
        print(ex)
        # response.status = 500
        return str(ex)


@route(path='/testquality', method=['OPTIONS', 'POST'])
@enable_cors
def test_quality():
    try:
        now = datetime.now()
        time_string = now.strftime("%H%M%S")

        calibration_data = request.json['calibration_data']
        calibration_data_df = pd.DataFrame(calibration_data)
        calibration_data_df.to_csv(results_output_folder_path +
                                   'test_quality_out_unsort' + time_string + '.csv', index=False)

        calibration_data_df = calibration_data_df.sort_values(by=['time'])

        calibration_data_df.to_csv(results_output_folder_path +
                                   'test_quality_out_' + time_string + '.csv', index=False)

        is_good, power_spectrum_mean, mean_sd_relation = check_signal_quality(calibration_data_df)
        freq = get_frequency_for_segment(calibration_data_df, 1000, 2000)

        graph_plt = get_x_data_plot(calibration_data_df, time_string)
        graph_plt.savefig(results_output_folder_path + 'chart_test_quality_out_' + time_string + '.png')

        pic_bytes = io.BytesIO()
        graph_plt.savefig(pic_bytes, format='png')
        pic_bytes.seek(0)
        base64_bytes = base64.b64encode(pic_bytes.read())
        base64_string = base64_bytes.decode('utf-8')
        graph_plt.close('all')

        response.content_type = 'application/json'
        output_data = dumps(
            {'is_good': is_good,
             'power_spectrum_mean': power_spectrum_mean,
             'mean_sd_relation': mean_sd_relation,
             'freq': str(freq),
             'image': base64_string})
        return output_data

    except Exception as ex:
        print(ex)
        # response.status = 500
        return str(ex)


@route(path='/getfacedataid', method=['OPTIONS', 'POST'])
@enable_cors
def get_face_data_id():
    try:
        base64_image_string = request.json['base64_image_string']
        image_id = request.json['image_id']
        cv_image = readb64(base64_image_string)
        face_boxes = classifiers.predict_face_position(cv_image)
        response.content_type = 'application/json'
        output_data = dumps(
            {'image_id': image_id,
             'face_boxes': face_boxes})
        return output_data

    except Exception as ex:
        print(ex)
        # response.status = 500
        return str(ex)


@route(path='/getfacedataandimg', method=['OPTIONS', 'POST'])
@enable_cors
def get_face_data_and_img():
    try:
        base64_image_string = request.json['base64_image_string']
        cv_image = readb64(base64_image_string)
        face_boxes = classifiers.predict_face_position(cv_image)
        face_image = apply_face_boxes(cv_image, face_boxes)

        pil_img = Image.fromarray(face_image)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        width, height = pil_img.size
        base_64_output_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")

        # base_64_output_image = base64.b64encode(face_image)
        response.content_type = 'application/json'
        output_data = dumps({'face_boxes': face_boxes,
                             'face_img_base_64': base_64_output_image_string})
        return output_data

    except Exception as ex:
        print(ex)
        # response.status = 500
        return str(ex)


def readb64(base64_string):
    encoded_data = base64_string.split(',')[1]
    decoded_data = base64.b64decode(encoded_data)
    sbuf = BytesIO()
    sbuf.write(decoded_data)
    pimg = Image.open(sbuf)
    pimg_arr = np.array(pimg)
    image = cv.cvtColor(pimg_arr, cv.COLOR_RGB2BGR)
    return image


def apply_face_boxes(image, face_boxes):
    for current_face_box in face_boxes:
        xmin = current_face_box[0]
        ymin = current_face_box[1]
        xmax = current_face_box[2]
        ymax = current_face_box[3]
        start_point = (xmin, ymin)
        end_point = (xmax, ymax)
        thickness = 3
        color = (255, 0, 0)
        image = cv.rectangle(image, start_point, end_point, color, thickness)
    return image


def write_debug_image(image, face_boxes):
    image = apply_face_boxes(image, face_boxes)
    cv.imwrite('D:\\DEVELOPMENT\\SaccadeOnlineClient\\test_image.jpg', image)


def init_classifiers():
    global classifiers
    classifiers = ClassificationManager()
    classifiers.initialize_models()



def get_frequency_for_segment(df, start_index, end_index):
    raw_time = df['time'].to_numpy()
    over_one = np.where(raw_time >= start_index)
    index_over_two = np.where(raw_time[over_one[0][0]:] >= end_index)
    tracker_frequency = index_over_two[0][0]
    return tracker_frequency


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

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def check_signal_quality(calibration_data):
    x_data = np.array(calibration_data['gaze_x'])
    f, P = periodogram(x_data)
    power_spectrum_mean = np.round(np.mean(P), 4)
    mean_sd_relation = abs(np.round(signaltonoise(x_data), 4))

    if power_spectrum_mean >= 0.1 or mean_sd_relation <= 1.0:
        return True, power_spectrum_mean, mean_sd_relation
    else:
        return False, power_spectrum_mean, mean_sd_relation

def get_x_data_plot(gaze_df, title):
    ax = gaze_df.dropna()[['gaze_x']].plot()
    ax.set_title(title)
    return plt

if __name__ == "__main__":
    print('starting')
    init_classifiers()
    run(host='brainonline.pja.edu.pl', port=8080, server='gevent', keyfile='C:\\DEV\\certificates\\cert_key.pem',
        certfile='C:\\DEV\\certificates\\brainonline_pja_edu_pl_cert.cer', reloader=1)
    # run(app, host='localhost', port=8080, debug=True, server='gevent')
