"""
Designed to run blind de-noising on 12-bit greyscale MRAW files. 
"""
from os import path
import argparse
import numpy as np
import warnings
import xmltodict
import bm4d
from scipy import fft
from scipy.interpolate import interp1d
import pyMRAW

SUPPORTED_FILE_FORMATS = ['mraw', 'tiff']
SUPPORTED_EFFECTIVE_BIT_SIDE = ['lower', 'higher']

#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MRAW processing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import 12-bit MRAW. Modified from pyMRAW software (which supports 8 or 16-bit only).
def get_cihx(filename):
    name, ext = path.splitext(filename)
    if ext == '.cihx':
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            first_last_line = [ i for i in range(len(lines)) if '<cih>' in lines[i] or '</cih>' in lines[i] ]
            xml = ''.join(lines[first_last_line[0]:first_last_line[-1]+1])

        raw_cih_dict = xmltodict.parse(xml)
        cih = {
            'Date': raw_cih_dict['cih']['fileInfo']['date'], 
            'Camera Type': raw_cih_dict['cih']['deviceInfo']['deviceName'],
            'Record Rate(fps)': float(raw_cih_dict['cih']['recordInfo']['recordRate']),
            'Shutter Speed(s)': float(raw_cih_dict['cih']['recordInfo']['shutterSpeed']),
            'Total Frame': int(raw_cih_dict['cih']['frameInfo']['totalFrame']),
            'Original Total Frame': int(raw_cih_dict['cih']['frameInfo']['recordedFrame']),
            'Image Width': int(raw_cih_dict['cih']['imageDataInfo']['resolution']['width']),
            'Image Height': int(raw_cih_dict['cih']['imageDataInfo']['resolution']['height']),
            'File Format': raw_cih_dict['cih']['imageFileInfo']['fileFormat'],
            'EffectiveBit Depth': int(raw_cih_dict['cih']['imageDataInfo']['effectiveBit']['depth']),
            'EffectiveBit Side': raw_cih_dict['cih']['imageDataInfo']['effectiveBit']['side'],
            'Color Bit': int(raw_cih_dict['cih']['imageDataInfo']['colorInfo']['bit']),
            'Comment Text': raw_cih_dict['cih']['basicInfo'].get('comment', ''),
        }

    else:
        raise Exception('Unsupported configuration file ({:s})!'.format(ext))

    # check exceptions
    ff = cih['File Format']
    if ff.lower() not in SUPPORTED_FILE_FORMATS:
        raise Exception('Unexpected File Format: {:g}.'.format(ff))
    bits = cih['Color Bit']
    if bits < 12:
        warnings.warn('Not 12bit ({:g} bits)! clipped values?'.format(bits))
                # - may cause overflow')
                # 12-bit values are spaced over the 16bit resolution - in case of photron filming at 12bit
                # this can be meanded by dividing images with //16
    if cih['EffectiveBit Depth'] != 12:
        warnings.warn('Not 12bit image!')
    ebs = cih['EffectiveBit Side']
    if ebs.lower() not in SUPPORTED_EFFECTIVE_BIT_SIDE:
        raise Exception('Unexpected EffectiveBit Side: {:g}'.format(ebs))
    if cih['Original Total Frame'] > cih['Total Frame']:
        warnings.warn('Clipped footage! (Total frame: {}, Original total frame: {})'.format(cih['Total Frame'], cih['Original Total Frame'] ))

    return cih

def read_uint12(data_chunk,count):
    # Solution shamelessley stolen from StackOverflow 
    # https://stackoverflow.com/questions/44735756/python-reading-12-bit-binary-files
    data = np.frombuffer(data_chunk,dtype=np.uint8,count=count*3//2)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])


def load_images(mraw, h, w, N, bit=12, roll_axis=True):
    """
    loads the next N images from the binary mraw file into a numpy array.
    Inputs:
        mraw: an opened binary .mraw file
        h: image height
        w: image width
        N: number of sequential images to be loaded
        roll_axis (bool): whether to roll the first axis of the output 
            to the back or not. Defaults to True
    Outputs:
        images: array of shape (h, w, N) if `roll_axis` is True, or (N, h, w) otherwise.
    """

    if int(bit) == 16:
        bit_dtype = np.uint16
    elif int(bit) == 8:
        bit_dtype = np.uint8
    elif int(bit) == 12:
        bit_dtype = None 
    else:
        raise Exception('Only 16-bit, 12-bit, and 8-bit files supported!')

    if bit_dtype is None:
        # 12-bit
        with open(mraw,"rb") as f:
            data = f.read()
            images = read_uint12(data,h*w*N).reshape(N,h,w)

    else:
        images = np.memmap(mraw, dtype=bit_dtype, mode='r', shape=(N, h, w))
    #images=np.fromfile(mraw, dtype=np.uint16, count=h * w * N).reshape(N, h, w) # about a 1/3 slower than memmap when loading to RAM. Also memmap doesn't need to read to RAM but can read from disc when needed.
    if roll_axis:
        return np.rollaxis(images, 0, 3)
    else:
        return images


def load_video(cih_file):
    """
    Loads and returns images and a cih info dict.
    
    Inputs:
    cih_filename: path to .cih or .cihx file, with a .mraw file 
        with the same name in the same folder.
    Outputs:
        images: image data array of shape (N, h, w)
        cih: cih info dict.

    """
    cih = get_cihx(cih_file)
    mraw_file = path.splitext(cih_file)[0] + '.mraw'
    N = cih['Total Frame']
    h = cih['Image Height']
    w = cih['Image Width']
    bit = cih['Color Bit']
    images = load_images(mraw_file, h, w, N, bit, roll_axis=False)
    return images, cih

class Transformer():
    def __init__(self,noise,clip):
        self.fit_rescale(clip)
        self.fit_mean(noise)

    def fit_rescale(self,frames):
        self.light = frames.max()
        self.dark = frames.min()
    
    def fit_mean(self,frames):
        self.mean = frames.mean()

    def transform(self,frames):
        return (frames-self.mean)/(self.light-self.dark)

    def scale(self,frames):
        tr_light = frames.min()
        tr_dark = frames.max()
        return self.dark + self.light*(frames-tr_dark)/(tr_light-tr_dark)


#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main code block ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cihx',help="Path to the .cihx file for the video. The corresponding .mraw file should be in the same directory.")
    parser.add_argument('x0',help="Start frame of section to sample noise from.",type=int)
    parser.add_argument('x1',help="End frame of section to sample noise from.",type=int)
    parser.add_argument('y0',help="Start frame of section of video to be denoised.",type=int)
    parser.add_argument('y1',help="End frame of section of video to be denoised.",type=int)
    parser.add_argument('--bm4d_profile',choices=["np","lc","refilter"],default="np",help="Profile option for bm4d.")

    args = parser.parse_args()

    # Load in mraw
    print("Loading MRAW... ",end="")
    images,cih = load_video(r"{}".format(args.cihx))
    noise_sample = images[args.x0:args.x1,:,:]
    clip = images[args.y0:args.y1,:,:]
    print("Done!")

    # Apply a rescaling
    print("Rescaling images... ",end="")
    transformer = Transformer(noise_sample,clip)
    noise_sample = transformer.transform(noise_sample)
    clip = transformer.transform(clip)
    print("Done!")

    # Calculate psd
    print("Calculating power spectrum density... ",end="")
    noise_ft = fft.fftn(noise_sample,workers=-1)
    psd_noise = np.imag(noise_ft)**2 + np.real(noise_ft)**2
    psd_resample = interp1d((args.y1-args.y0-1)/(args.x1-args.x0-1)*np.arange(args.x1-args.x0),psd_noise,axis=0,copy=False)(np.arange(args.y1-args.y0))*np.prod(clip.shape)
    print("Done!")

    print("Applying bm4d algorithm... ",end="")
    clip_denoised = bm4d.bm4d(clip,psd_resample,profile=args.bm4d_profile)
    print("Done!")

    clip_out = (transformer.scale(clip_denoised)).astype(images.dtype)

    # Save a denoised mraw file.  
    file_out = args.cihx.split(".")
    file_out[-2] += "_denoised"
    file_out = ".".join(file_out)
    pyMRAW.save_mraw(clip_out,file_out,bit_depth=16,ext="mraw",
                     info_dict={"Comment Text":"Clip denoised using bm4d algorithm.",
                                "Record Rate(fps)":cih.get("Record Rate(fps)"),
                                "Shutter Speed(s)":cih.get("Shutter Speed(s)")})
    print("Denoised MRAW saved to {}".format(file_out))
