import argparse
import librosa
import numpy
from PIL import Image,ImageDraw


def main():
    parser=argparse.ArgumentParser(prog='synical',description='Create images from audio')
    parser.add_argument("input",help='Path to audio file')
    parser.add_argument("-o", "--output", default="wallpaper.png", help="Path to save the generated image")
    parser.add_argument("-s", "--size", type=int, default=1080, help="Vertical resolution of the wallpaper")
    args=parser.parse_args()
    print('Loading file')
    audio,sr=librosa.load(args.input,duration=10)
    print('Loading mfcc')
    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=21)
    norm=((mfcc-numpy.min(mfcc))/(numpy.max(mfcc)-numpy.min(mfcc)))*255
    step=0
    #Image Generation
    oldrgb=numpy.array([128,128,128])
    width=norm.shape[1]
    img=Image.new('RGB',(width,args.size))
    draw=ImageDraw.Draw(img)
    for i in range(width):
        newrgb_data = numpy.array([norm[0, i], norm[1, i], norm[2, i]])
        current_rgb = (oldrgb * 0.9) + (newrgb_data * 0.1)
        pixel_color = numpy.clip(current_rgb, 0, 255).astype(numpy.uint8)
        draw.line([(i, 0), (i, args.size)], fill=tuple(pixel_color))
        oldrgb = current_rgb # Keep as float for next iteration
    
    final_wallpaper = img.resize((1920, args.size), Image.Resampling.LANCZOS)
    final_wallpaper.save(args.output)


        
        
    






if __name__ == "__main__":
    main()
