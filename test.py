# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import os
from utils import parse_musescore
from pitchDetection.mfshs import MFSHS
from utils import parse_musescore, get_wav_and_json_file, save_files, draw_array
from utils import post_cnn_onset
from alignment import sw_alignment
from onset_predict import predictor_onset
from post_process import Evaluator, trans_onset_and_offset

import numpy as np


def get_files():
    test_dir = './data/test1'
    wav_files = [os.path.join(test_dir, x) for x in os.listdir(
        test_dir) if x.endswith('wav') or x.endswith('mp3')]

    txt_files = [os.path.join(x[0:-3] + 'txt') for x in wav_files]
    return wav_files, txt_files


def get_gound_score_notes(txt_files):
    ground_notes, score_notes = dict(), dict()
    for file in txt_files:
        ground_note, score_note = [], []
        with open(file, 'r') as fr:
            fr.readline()
            lines = fr.readlines()
            for line in lines:
                line = line.strip().split('\t')
                ground_note.append(float(line[0]))
                score_note.append(float(line[1]))

        file_id = os.path.basename(file)[0:-4]
        ground_notes[file_id] = ground_note
        score_notes[file_id] = score_note

    return ground_notes, score_notes


def is_octive(det_note, score_note):
    det_note = np.array(det_note)
    score_note = np.array(score_note)
    diff_note = det_note - score_note
    octive1 = np.where((diff_note > 10) & (diff_note < 14))[0]
    octive2 = np.where((diff_note > -14) & (diff_note < -10))[0]
    is_octive1 = (len(octive1) / len(diff_note)) > 0.8
    is_octive2 = (len(octive2) / len(diff_note)) > 0.8

    if is_octive1:
        det_note = det_note - 12
    elif is_octive2:
        det_note = det_note + 12
    return det_note


def cal_acc(det_note, score_note):
    det_note = np.array(det_note)
    det_note = is_octive(det_note, score_note)
    score_note = np.array(score_note)
    diff_note = abs(det_note - score_note)
    acc_index = np.where(diff_note <= 0.5)[0]
    accuracy = len(acc_index) / len(diff_note)
    return accuracy, diff_note


def print_save_acc(wav_files, ground_notes, score_notes):
    fwt = open('log/acc.txt', 'w')
    for index in range(len(wav_files)):
        json_path = '/home/data/lj/onset_detect/MUS/evaluate/'
        wav_file = wav_files[index]
        fid = os.path.basename(wav_file)[0:-4]
        # score_note = np.array(score_notes[fid]).astype(int)
        json_path = os.path.join(json_path, fid)
        score_file = [os.path.join(json_path, x) for x in os.listdir(
            json_path) if x.endswith('json')][0]

        mfshs = MFSHS(wav_file)
        mfshs.pitch_detector()
        pitches = mfshs.pitches
        zero_amp_frame = mfshs.zeroAmploc

        score_note, note_types, pauseLoc = parse_musescore(
            score_file)  # parse musescore
        predictor = predictor_onset()
        onset_time = predictor.predict(wav_file)

        onset_frame = predictor.onset_frame

        onset_frame = post_cnn_onset(pitches, onset_frame)

        match_loc_info = sw_alignment(pitches,
                                      onset_frame,
                                      score_note)

        onset_offset_pitches = trans_onset_and_offset(match_loc_info,
                                                      onset_frame,
                                                      pitches)
        filename_json = os.path.splitext(wav_file)[0] + ".json"
        evaluator = Evaluator(filename_json,
                              onset_offset_pitches,
                              zero_amp_frame,
                              score_note,
                              pauseLoc,
                              note_types)

        print(wav_file)
        sys_acc, sys_diff_note = cal_acc(evaluator.det_note, score_note)

        gnote = ground_notes[fid]
        snote = score_note
        ground_acc, ground_diff_note = cal_acc(gnote, snote)

        ground_sys_acc, ground_sys_diff_note = cal_acc(
            evaluator.det_note, gnote)

        strs = '{}\t{}\t{}'.format('ground', 'sys', 'ground_sys')
        print(strs)
        strs = '{:.3f}\t{:.3f}\t{:.3f}\n'.format(
            ground_acc, sys_acc, ground_sys_acc)
        print(strs)

        file_name = os.path.join('log', fid + '.txt')
        with open(file_name, 'w') as fw:
            fw.write('gnote\t\tsnote\t\tdnote\n')
            for index in xrange(len(gnote)):
                fw.write('{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t'.format(
                    gnote[index], snote[index], evaluator.det_note[index]))

                fw.write('{:.3f}\t\t{:.3f}\t\t{:.3f}\n'.format(ground_diff_note[
                         index], sys_diff_note[index], ground_sys_diff_note[index]))

            fw.write('{:.3f}\t\t{:.3f}\t\t{:.3f}\n'.format(
                ground_acc, sys_acc, ground_sys_acc))

        fwt.write(wav_file + '\n')
        fwt.write(strs)
        fwt.write('\n')
        fwt.flush()
    fwt.close()

if __name__ == '__main__':
    wav_files, txt_files = get_files()
    ground_notes, score_notes = get_gound_score_notes(txt_files)
    print_save_acc(wav_files, ground_notes, score_notes)
