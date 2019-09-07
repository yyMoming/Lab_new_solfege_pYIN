# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import os
from utils import parse_musescore
from pitchDetection.mfshs import MFSHS
from utils import parse_musescore, get_wav_and_json_file, save_files,draw_array
from utils import post_cnn_onset
from alignment import sw_alignment
from onset_predict import predictor_onset
from post_process import Evaluator, trans_onset_and_offset
from pypYIN import demo
import numpy as np
import json

def main(wav_file, score_file):
    # mfshs = MFSHS(wav_file)
    # mfshs.pitch_detector()
    # pitches = mfshs.pitches #Ã¨Â¿â€Ã¥â€ºÅ¾Ã©Å¸Â³Ã©Â«Ë?
    # print(type(pitches),pitches)
    # zero_amp_frame = mfshs.zeroAmploc   #Ã©Å¸Â³Ã©Â«ËœÃ¤Â¸?Ã§Â´Â¢Ã¥Â¼â€?
    # print(type(zero_amp_frame),zero_amp_frame)
    
    # pitches_filepath = "/home/ywm/MUSIC/new_solfege_pYIN/data/1011_pitch.txt"
    # pitches = []
    # with open(pitches_filepath,'r') as f:
    #   a = f.readlines()
    #   for i in a:
    #     pitches.append(float(i.split()[0]))
    # pitches = np.array(pitches)

    pitches = demo.pYIN(wav_file)
    pitches = np.array(pitches) - 20
    pitches = np.where((pitches<0.0),0,pitches)

    #print(type(pitches),pitches)
    zero_amp_frame = np.where(pitches==0)[0]
    score_note, note_types, pauseLoc = parse_musescore( #Ã¨Â§Â£Ã¦Å¾ÂjsonÃ¤Â¹ÂÃ¨Â°Â±,Ã¨Â¿â€Ã¥â€ºÅ¾Ã¤Â¹ÂÃ¨Â°Â±Ã¤Â¸Â­Ã§Å¡â€žnoteÃ¥â‚¬Â¼Ã¥â€™Å’Ã¤Â¼â€˜Ã¦Â­Â¢Ã§Â¬Â¦Ã¤Â½ÂÃ§Â?
        score_file)  # parse musescore

    predictor = predictor_onset()
    onset_time = predictor.predict(wav_file)

    #draw_array(predictor.onset_pred)
    onset_frame = predictor.onset_frame
    
    onset_frame = post_cnn_onset(pitches,onset_frame)
    #print("onset_frame:",onset_frame)
    match_loc_info = sw_alignment(pitches,
                                  onset_frame,
                                  score_note)
    #print(2)
    onset_offset_pitches = trans_onset_and_offset(match_loc_info,
                                                  onset_frame,
                                                  pitches)
    #print("onset_offset_pitches:",onset_offset_pitches)
    filename_json = os.path.splitext(wav_file)[0] + ".json"
    evaluator = Evaluator(filename_json,
                          onset_offset_pitches,	
                          zero_amp_frame,
                          score_note,
                          pauseLoc,
                          note_types)
    #print(4)
    save_files(wav_file,
               onset_frame,
               pitches,
               evaluator.det_note,
               score_note,
               onset_offset_pitches['onset_frame'])

    #print(5)
    return evaluator.score
if __name__ == '__main__':
    root_path = os.path.join(os.path.dirname(__file__), 'data')
    wav_files, score_json = [], []
    score_dict = {}
    get_wav_and_json_file(root_path, wav_files, score_json)
    for i in range(len(wav_files)):
        cur_score = main(wav_files[i], score_json[i])
        score_dict[os.path.splitext(wav_files[i])[0]] = cur_score

    with open(os.path.join(root_path,'all_score.json'),'w+') as f:
        json.dump(score_dict,f)
        print("--------")


