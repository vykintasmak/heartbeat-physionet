"""
Created by Vykinitas Maknickas (vykintas.mak[at]gmail.com)
LinkedIn: https://lt.linkedin.com/in/vykintasmak

@author: Vykintas Maknickas

This code is required to prepare data for learning Normal/Abnormal heart rates from .wav files

"""

from __future__ import absolute_import
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import scipy.signal
from numpy import inf
import random
import math


def load_data(normal_path='',
              abnormal_path='',
              test_split=0.2, 
              seed=113,  
              width=128, 
              height=128, 
              randomize=True, 
              category_split_equal=True):

        # reading files from folders
        normal_files = [f for f in listdir(normal_path) if isfile(join(normal_path, f))]
        murmur_files = [f for f in listdir(abnormal_path) if isfile(join(abnormal_path, f))]

        # randomizing files
        random.seed(seed)
        random.shuffle(normal_files)
        random.seed(seed)
        random.shuffle(murmur_files)

        # splitting files into train and validation sets
        normal_files_test = normal_files[:int(len(normal_files) * (1 - test_split))]
        normal_files_validation = normal_files[int(len(normal_files) * (1 - test_split)):]

        murmur_files_test =  murmur_files[:int(len(murmur_files) * (1 - test_split))]
        murmur_files_validation = murmur_files[int(len(murmur_files) * (1 - test_split)):]

        normal_len_test = 0
        murmur_len_test = 0

        normal_len_validation = 0
        murmur_len_validation = 0


        # counting the number of frames in each set for an equal split of normal and abnormal set in training set if category_split_equal=True
        for files in normal_files_test:
                normal_len_test += get_numresults(audiopath=normal_path+files, length=width, binsize=height)

        for files in murmur_files_test:
                murmur_len_test += get_numresults(audiopath=abnormal_path+files, length=width, binsize=height)

        for files in normal_files_validation:
                normal_len_validation += get_numresults(audiopath=normal_path+files, length=width, binsize=height)

        for files in murmur_files_validation:
                murmur_len_validation += get_numresults(audiopath=abnormal_path+files, length=width, binsize=height)

        print("Normal frames test:"+str(normal_len_test))
        print("Abnormal frames test:"+str(murmur_len_test))
        print("Normal frames validation:"+str(normal_len_validation))
        print("Abnormal frames test:"+str(murmur_len_validation))

        # setting the equal number of normal/abnormal frames if category_split_equal=True
        if normal_len_test > murmur_len_test and category_split_equal:
                max_len_normal_test = murmur_len_test
                max_len_murmur_test = murmur_len_test
        elif normal_len_test <= murmur_len_test and category_split_equal:
                max_len_normal_test = normal_len_test
                max_len_murmur_test = normal_len_test
        elif not category_split_equal:
                max_len_normal_test = normal_len_test
                max_len_murmur_test = murmur_len_test

        if normal_len_validation > murmur_len_validation and category_split_equal:
                max_len_normal_validation = murmur_len_validation
                max_len_murmur_validation = murmur_len_validation
        elif normal_len_validation <= murmur_len_validation and category_split_equal:
                max_len_normal_validation = normal_len_validation
                max_len_murmur_validation = normal_len_validation
        elif not category_split_equal:
                max_len_normal_validation = normal_len_validation
                max_len_murmur_validation = murmur_len_validation

        total_len_test = max_len_normal_test + max_len_murmur_test

        total_len_validation = max_len_normal_validation + max_len_murmur_validation

        # getting the shape of a single frame
        first_mels = mels(audiopath=normal_path+normal_files_test[0], length=width, binsize=height)
        w, h = first_mels[0].shape

        X = np.zeros((total_len_test, 3, w, h))
        Y = np.zeros((total_len_validation, 3, w, h))
        labels_test = np.zeros((total_len_test, 1), dtype=np.int8)
        labels_val = np.zeros((total_len_validation, 1), dtype=np.int8)
        
        # filling training set array with frames from Normal files
        # frames cosists of mels, their deltas and delta-deltas
        file_num = 0
        z = 0
        for filename in normal_files_test:
                full_path = normal_path + filename
                result = mels(audiopath=full_path, length=width, binsize=height)
                result_delta = mels(audiopath=full_path, length=width, deltas=1, binsize=height)
                result_deltadelta = mels(audiopath=full_path, length=width, deltas=2, binsize=height)
                for add, add_delta, add_deltadelta in zip(result, result_delta, result_deltadelta):
                        X[file_num][0] = add
                        X[file_num][1] = add_delta
                        X[file_num][2] = add_deltadelta
                        labels_test[file_num] = 0
                        file_num += 1
                        z += 1
                        if z >= max_len_normal_test:
                                break


                if z >= max_len_normal_test:
                        break
        
        # filling validations set array with frames from Normal files
        # frames cosists of mels, their deltas and delta-deltas        
        z = 0
        file_num_validation = 0
        for filename in normal_files_validation:
                full_path = normal_path + filename
                result = mels(audiopath=full_path, length=width, binsize=height)
                result_delta = mels(audiopath=full_path, length=width, deltas=1, binsize=height)
                result_deltadelta = mels(audiopath=full_path, length=width, deltas=2, binsize=height)
                for add, add_delta, add_deltadelta in zip(result, result_delta, result_deltadelta):
                        Y[file_num_validation][0] = add
                        Y[file_num_validation][1] = add_delta
                        Y[file_num_validation][2] = add_deltadelta
                        labels_val[file_num_validation] = 0
                        file_num_validation += 1
                        z += 1
                        if z >= max_len_normal_validation:
                                break

                if z >= max_len_normal_validation:
                        break

        # filling training set array with frames from Abnormal files
        # frames cosists of mels, their deltas and delta-deltas
        z = 0
        for filename in murmur_files_test:
                full_path = abnormal_path + filename
                result = mels(audiopath=full_path, length=width, binsize=height)
                result_delta = mels(audiopath=full_path, length=width, deltas=1, binsize=height)
                result_deltadelta = mels(audiopath=full_path, length=width, deltas=2, binsize=height)
                for add, add_delta, add_deltadelta in zip(result, result_delta, result_deltadelta):
                        X[file_num][0] = add
                        X[file_num][1] = add_delta
                        X[file_num][2] = add_deltadelta
                        labels_test[file_num] = 1
                        file_num += 1
                        z += 1
                        if z >= max_len_murmur_test:
                                break

                if z >= max_len_murmur_test:
                        break

        # filling validation set array with frames from Abnormal files
        # frames cosists of mels, their deltas and delta-deltas
        z = 0
        for filename in murmur_files_validation:
                full_path = abnormal_path + filename
                result = mels(audiopath=full_path, length=width, binsize=height)
                for add, add_delta, add_deltadelta in zip(result, result_delta, result_deltadelta):
                        Y[file_num_validation][0] = add
                        Y[file_num_validation][1] = add_delta
                        Y[file_num_validation][2] = add_deltadelta
                        labels_val[file_num_validation] = 1
                        file_num_validation += 1
                        z += 1
                        if z >= max_len_murmur_validation:
                                break

                if z >= max_len_murmur_validation:
                        break


        # randomizing test and validation sets, using seed for reproducability 
        if randomize:
                np.random.seed(seed)
                np.random.shuffle(X)
                np.random.seed(seed)
                np.random.shuffle(labels_test)
                np.random.seed(seed)
                np.random.shuffle(Y)
                np.random.seed(seed)
                np.random.shuffle(labels_val)

        X_train = X
        y_train = labels_test
        X_test = Y
        y_test = labels_val

        return (X_train, y_train), (X_test, y_test)

def data_from_file(filename, max_frames=2, width=129, height=256):

    first_mels = mels(audiopath=filename, length=width, binsize=height)

    w, h = first_mels[0].shape

    if len(first_mels) < max_frames:
        X = np.zeros((len(first_mels), 3, w, h))
    else:
        X = np.zeros((max_frames, 3, w, h))

    file_num = 0

    result = first_mels
    result_delta = mels(audiopath=filename, length=width, deltas=1, binsize=height)
    result_deltadelta = mels(audiopath=filename, length=width, deltas=2, binsize=height)

    for add, add_delta, add_deltadelta in zip(result, result_delta, result_deltadelta):
        X[file_num][0] = add
        X[file_num][1] = add_delta
        X[file_num][2] = add_deltadelta

        file_num += 1
        if file_num >= max_frames:
            break
             
    return X

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    

def normalized(a): 
        
    a = (a - a.min())/(a.max()-a.min())
    
    return a
 
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs

def get_numresults(audiopath, length=1024, binsize=256):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    ims[ims == -inf] = 0
    ims[ims == inf] = 0

    ims = np.transpose(ims)

    freqbins, timebins = np.shape(ims)
    num_results = math.ceil(float(timebins)/float(length))

    return num_results

def mels(audiopath, length=1024, binsize=256, deltas=0):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    ims[ims == -inf] = 0
    ims[ims == inf] = 0

    ims = np.transpose(ims)

    if deltas > 0:
        ims = delta(ims, order=deltas)

    freqbins, timebins = np.shape(ims)

    melss = np.array(ims)

    num_results = math.ceil(float(timebins)/float(length))

    result = np.zeros((num_results, freqbins, length))
    
    start = 0
    end = length
    for z in range(int(num_results)):
        x = 0
        for q in melss:
            if length > len(q[start:end]):
                zeros = np.zeros((length-len(q[start:end])))
                q = np.concatenate((q, zeros))
            result[z][x] = q[start:end]
            x += 1
        zero_bool = result[z] == 0
        result[z][zero_bool] = 1e-5

        start += length
        end += length

    return normalized(result)
def delta(data, width=9, order=1, axis=-1, trim=True):
    data = np.atleast_1d(data)

    if width < 3 or np.mod(width, 2) != 1:
        print('width must be an odd integer >= 3')

    if order <= 0 or not isinstance(order, int):
        print('order must be a positive integer')

    half_length = 1 + int(width // 2)
    window = np.arange(half_length - 1., -half_length, -1.)

    # Normalize the window so we're scale-invariant
    window /= np.sum(np.abs(window)**2)

    # Pad out the data by repeating the border values (delta=0)
    padding = [(0, 0)] * data.ndim
    width = int(width)
    padding[axis] = (width, width)
    delta_x = np.pad(data, padding, mode='edge')

    for _ in range(order):
        delta_x = scipy.signal.lfilter(window, 1, delta_x, axis=axis)

    # Cut back to the original shape of the input data
    if trim:
        idx = [slice(None)] * delta_x.ndim
        idx[axis] = slice(- half_length - data.shape[axis], - half_length)
        delta_x = delta_x[idx]

    return delta_x

