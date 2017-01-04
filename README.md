# Classification of Normal/Abnormal Heart Sound Recordings
The algorithm ranked 6th in the PhysioNet/Computing in Cardiology Challenge 2016
Sensitivity: 0.8063, Specitivity: 0.8766, Overall challenge score: 0.8415

# Requirements
Python 2.7
Keras 1.0.1

To try out the code you need to create two separate folders for Normal and Abnormal .wav files and insert them into hb_prediction_v2.py 
normal_folder='',  abnormal_folder=''

Example usage:
python hb_prediction_v2.py -o train
python hb_prediction_v2.py -o test -i filename


