'''
Created on Fri Nov 10 11:15:55 2017

@author: Muhammad Shahnawaz

This script is to perform the preprocessing of the ECS-50 dataset present in the 'Samples/' directory.
In this scrip we read the files and compute the mel_spectrogram of these files the results are saved in 'Preproc/mfcc/fileName.npy' files.
'''
# importing the dependencies
import librosa, os, numpy as np

''' get all the class names, which are the subdirectories in the path = 'Samples/' directory,
default value is 'Samples/', pass your directory path
Each subdirectory has all the files for the belonging class with subdirectory name as the eventType. '''


def get_class_names(path='Samples/'):
    class_names = os.listdir(path)
    return class_names


# Main function starts here
def main(inpath='Samples/', outpath='Preproc/'):
    # defining some variables to be used later
    nMels = 128
    # hopLength = 1323
    # numFrames = 120
    # deleting the files of preprocessed data if already exist
    if(os.path.isfile('X_test.npy')):
        os.remove('X_test.npy')
    if(os.path.isfile('Y_test.npy')):
        os.remove('Y_test.npy')
    if(os.path.isfile('X_train.npy')):
        os.remove('X_train.npy')
    if(os.path.isfile('Y_train.npy')):
        os.remove('Y_train.npy')
    if not os.path.exists(inpath):
        print('The inptut directory doesnot exist!')
        return 0
    if not os.path.exists(outpath):   # if there is no directory for outputh file
        os.mkdir( outpath, 0o755 );   # make a new directory for preproced files
    # getting all the class names
    class_names = get_class_names(path=inpath)
    print('There are {:3d} total classes. Preprocessing class by class:'.format(len(class_names)))
    for idx, classname in enumerate(class_names):   # go through the subdirs
        if not os.path.exists(outpath+classname):
            os.mkdir( outpath+classname, 0o755 );   # make a new subdirectory for preproc class
        class_files = os.listdir(inpath+classname) # get the name of all the files in the class subdirectory
        n_files = len(class_files) # number of files
        print(' Preprocessing  {:2d} files for class {:s}.'.format(n_files,classname)) # verbose

        for idx2, infilename in enumerate(class_files):
            audio_path = inpath + classname + '/' + infilename # constructing file name
            # print(audio_path)
            sr = 16000
            aud, sr = librosa.load(audio_path, sr=sr) # reading the content of the files
            # aud = librosa.resample(aud, sr, target_sr, res_type="kaiser_fast")
            # (S, ref=1.0, amin=1e-05, top_db=80.0)

            # melgram = librosa.power_to_db(librosa.feature.melspectrogram(aud, sr=sr, n_mels=nMels), ref=1.0)
            # melgram = melgram[np.newaxis,:,0:300,np.newaxis]

            frame_length = 0.1
            frame_stride = 0.01
            input_stride = int(round(sr*frame_stride))
            input_nfft = int(round(sr*frame_length))
            fmin = 0
            fmax = 8000

            # melgram = librosa.feature.melspectrogram(aud, sr=sr, n_mels=nMels)
            melgram = librosa.feature.melspectrogram(aud,
                                                     n_mels=128,
                                                     n_fft=input_nfft,
                                                     hop_length=input_stride,
                                                     sr=sr,
                                                     fmin=fmin,
                                                     fmax=fmax)
            melgram = melgram[np.newaxis, :, 0:300, np.newaxis]
            # print("Shape: {}".format(melgram.shape))
            outfile = outpath + classname + '/' + infilename[0:-4]+'.npy'  # creating a string name for out file names
            # if melgram.shape[2] > 155:
            np.save(outfile, melgram) # saving mfcc of the audio file in an *.npy file.
    return 0


# code entry point
if __name__ == '__main__':
    main()
