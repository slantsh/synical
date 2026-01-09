import argparse
import librosa
import numpy

def main():
    parser=argparse.ArgumentParser(prog='synical',description='Create images from audio')
    parser.add_argument("input",help='Path to audio file')
    parser.add_argument("-o", "--output", default="wallpaper.png", help="Path to save the generated image")
    parser.add_argument("-s", "--size", type=int, default=1080, help="Vertical resolution of the wallpaper")
    args=parser.parse_args()
    print('Loading file')
    audio,sr=librosa.load(args.input)
    print('Loading mfcc')
    mfcc=librosa.feature.mfcc(y=audio,sr=sr)
    with open('text.txt','w') as f:
        f.write(((numpy.array2string(mfcc,threshold=numpy.inf))))




if __name__ == "__main__":
    main()
