#-*- coding:utf-8 -*-
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os 
import math
import signal
import librosa
import numpy as np 
from config import pitch_det_param
from multiprocessing import Process,cpu_count,Manager

import time 

fftLength = pitch_det_param['fftLength']
windowLength = pitch_det_param['windowLength']
sampleRate = pitch_det_param['sampleRate']
frameSize = pitch_det_param['frameSize']
h = pitch_det_param['h']
H = pitch_det_param['H']
hopSize = pitch_det_param['hopSize']
hammingWindow = np.zeros(windowLength)


def callHamming():
    ''' 
        calculate 汉明窗
    '''
    global hammingWindow
    for i in range(0,windowLength):
        hammingWindow[i] = float(0.54-0.46*math.cos(2*math.pi*i/(windowLength-1)))

class MFSHS(object):
    '''
        use mfshs algorithm detect audio pitches
        param:
            wav_file audio file
    '''
    def __init__(self,wav_file):
        data_wav,fs_wav = librosa.load(wav_file,sr=sampleRate) #音频处理加载，浮点数时间序列，data_wav返回序列，fs_wav返回采样频率
        manager = Manager()#Manager（）控制一个服务器进程，该进程保存Python对象并允许其他进程使用代理操作它们
        self.pitch = manager.dict()
        self.energe = manager.dict()
        self.zeroamploc = manager.list()

        self.process_num = cpu_count()#返回系统中的CPU数量。
        y = np.zeros(frameSize//2)
        x = np.hstack([y,data_wav,y])#np.hstack((a,b))水平一维
        nFrame = np.floor((len(x)-frameSize)/hopSize)+1 #numpy.floor(x)不大于x的最大整数，nFrame总帧个数
        self.nFrame = int(nFrame)
        self.xFrame = np.zeros([self.nFrame,frameSize])#纵轴是帧数，横轴是帧长
        curPos = 0
        callHamming()
        for index in xrange(self.nFrame):
            self.xFrame[index,:] = x[curPos:curPos+frameSize]
            curPos = curPos+hopSize

    def pitch_detector(self):
        '''
            multi process 音高检测
            default process_num is system core
        '''
        process_list = []
        for index in xrange(self.process_num):
            p = Process(target=self.run,args=(self.nFrame*index//self.process_num,
                self.nFrame*(index+1)//self.process_num))
            p.start()
            process_list.append(p)
        for process in process_list:
            process.join()


    def run(self,start,end):
        for index in range(start,end):
            meanAmp = np.mean(np.abs(self.xFrame[index,:]))#每一进程中每一帧帧中各“采样点绝对值”的“均值”
            note = self.getNode(self.xFrame[index,:]) if meanAmp>0.01 else 0 #得到音高音符
            self.pitch[index] = note
            self.energe[index] = meanAmp
            if meanAmp<0.01:
                self.zeroamploc.append(index)


    def getNode(self,data):
        '''
            pitch detect interface
        '''
        fPitchResult = self.calculatePitcher(data)
        fPitchResult = 0 if fPitchResult <= 50 else fPitchResult
        fNote = (69+12*math.log(fPitchResult/440)/math.log(2)) if fPitchResult > 0 else 0
        fNote = (fNote-20) if (fNote > 0) else fNote
        return fNote

    def calculatePitcher(self,rawMicDat):
        fftResult = np.zeros(int(fftLength/2))  #4096
        allFFTResult = np.zeros(fftLength)  #8192
        allFFTResult[0:windowLength] = rawMicDat*hammingWindow  #2048
        fftResultNoPhase = np.fft.fft(allFFTResult) #2048
        fftResultNoPhase = np.abs(fftResultNoPhase) #2048
        fftResult[0:int(fftLength/2)] = np.zeros(int(fftLength/2))  #4096
        fftResult[0:int(fftLength/8)] = fftResultNoPhase[0:int(fftLength/8)]  #1024
        return self.calculateMFSHPitch(fftResult)

    def calculateMFSHPitch(self,fftResult):#置信度计算和返回基频音高频率
        maxResultIndex = (np.where(fftResult == max(fftResult)))[0][0]#最大FFT振幅索引
        p = np.zeros(H) #5
        for i in range(0,H):
            p[i] = 0
            L = min(10, int(math.floor((i+1)*fftLength/8/(maxResultIndex+1))))
            for j in range(0,L-1):
                if round((j+1)*(maxResultIndex+1)/(i+1)) != 0:
                    p[i] += fftResult[int(round((j+1)*(maxResultIndex+1)/(i+1))-1)]*pow(h,j)
        maxPIndex = (np.where(p == max(p)))[0][0]
        f0 = round((maxResultIndex+1)/(maxPIndex+1))/fftLength
        if f0 > 1100.0/sampleRate:
            f0 = 0
        fPitch = f0*sampleRate
        return fPitch

    @property
    def pitches(self):
        '''
            return pitch
        '''
        return np.round(np.array(self.pitch.values()),2) #小数点后面保留2位
    
    @property
    def energes(self):
        ''' 
            return energe
        '''
        return np.round(np.array(self.energe.values()),3)   #小数点后面保留3位

    @property
    def zeroAmploc(self):
        '''
            过零点位置
        '''
        return np.array(sorted(self.zeroamploc))

    def saveArray(self,filename,Array_list):
        with open(filename,"w") as f:
            for arr in Array_list:
                f.write(str(arr)+"\n")
