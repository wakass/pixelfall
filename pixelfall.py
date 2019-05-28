#!/usr/bin/env python3

__author__ = 'Chris WakA'
__version__ = '1.0'
__copyright__ = 'Copyright 2018 by Chris WakA'
__license__ = 'MIT'

import argparse
import sys
from os import path
import ipdb

from PIL import Image,ImagePalette
import numpy as np
from functools import reduce

DEBUG=False


pal_modes = {
#color cycle array for sequence 1,2,3,4,5
    'cycle' :
    {'x': np.array([[0, 0, 2, 1, 3, 2, 3]]),  #for 1,2,3,4,5 but pad for index 0, and index 6, to map to value 0,6 respectively
     'y': np.array([[0, 1, 0, 2, 1, 3, 3]]),
    'pal': np.array( [
[[0, 0, 2, 3],
 [0, 1, 3, 3]],
    
[[2, 0, 1, 0],
 [0, 3, 2, 3]],
    
[[2, 0, 2, 0],
 [3, 2, 1, 1]],
    
[[2, 2, 1, 0,],
 [3, 1, 3, 1]],
    
[[0, 1, 0, 3],
 [3, 2, 3, 1]]
])
    },
    'extend': {
    'x':np.array([[0, 0, 0, 0, 3, 3, 3]]),
    'y':np.array([[0, 1, 2, 3, 1, 2, 3]]),
    'pal': np.array([  [[0, 0, 0, 3], [0, 1, 2, 3]]  ])
    }
}

def make_pal_byteready(pal):
    pal = np.flipud(pal) #assumes color cycle will be read out back to front.
    pal = pal.reshape(-1,4)
    pal = np.fliplr(pal) #msb and order of palette is reversed
    reverse_map = np.array([[3,2,1,0]]) #switch around 0..3, to 3..0, because gb palette is darkest 3, lightest 0
    pal = reverse_map[np.arange(reverse_map.shape[0])[:,None],pal]
    
    pal_byteready = np.apply_along_axis(lambda z: reduce(lambda x,y: (y + (x<<2)),z),1,pal).astype('uint8')
    return pal_byteready


def main(file,output_path='./',pal_mode='cycle'):
    im = Image.open(file)
    base = path.basename(file)
    base,ext = path.splitext(base)

    #From 0,1,2,3 (regular gb color codes, encoded in 2bit)
    #we can get through frame over frame averaging: 7 colors
    #these will be (sum): 0,1,2,3,4,5,6 

    #let's reinterpret the grayscale image in png with indexed palette
    #to this new palette (eventually when PIL storts their stuff out)
    #https://github.com/python-pillow/Pillow/issues/2797
    rgb = np.array([0,1,2,3,4,5,6])*(255./7.)
    rgb = rgb.astype('uint8')
    neogb_palette = np.concatenate([[r,g,b] for r,g,b in (zip(rgb,rgb,rgb))])
    neogb_palette = tuple(neogb_palette) + (0,0,0)*(256-7)
    #Encoding in two split images:
    # high encodes. 0,1,2,3
    # low encodes. only 3 or 0

    # e.g.
    #         6 5 4 3 2 1 0
    # high    3 2 1 0 2 1 0
    # low     3 3 3 3 0 0 0 

    #6 is black <--linear--> 0 is white

    if not ((im.width % 8 == 0) and (im.height % 8 == 0)):
        raise ValueError("Image resolution only supports multiples of 8")
    
    im_pal = Image.new('P',(0,0),0)
    im_pal.putpalette(neogb_palette)
    im_pal.im.putpalette(*im_pal.palette.getdata())

    im_gb = im.convert(mode='RGB')
    im_gb = im_gb.quantize(colors=7,method=3,dither=False,palette=im_pal)
    im_gb.info['transparency'] = 0
    # im_gb.putpalette(neogb_palette)
    # im_gb.im.putpalette(*im_gb.palette.getdata())
    if DEBUG:
        im_gb.save(output_path + 'base_interm.gif')

    def split_lohi(img,x,y):
        im = np.frombuffer(img.tobytes(),dtype='uint8')


        #according to experiments in palette permutations [ref. #notpublished] (see above)
        #e.g. the following [1,2,3,4,5] discombobulates nicely into:
        # x: [0 2 1 3 2]
        # y: [1 0 2 1 3] 
        im_low = x[np.arange(x.shape[0])[:,None], im]
        im_high = y[np.arange(y.shape[0])[:,None], im]


        im_low = im_low.astype('uint8')
        im_high =  im_high.astype('uint8')


        im_lowt = img.copy()
        im_hight = img.copy()

        im_lowt.frombytes(im_low.tobytes())
        im_hight.frombytes(im_high.tobytes())
        return im_lowt,im_hight

    pal = pal_modes[pal_mode]
    im_low,im_high= split_lohi(im_gb, pal['x'],pal['y'])
    pal_byteready = make_pal_byteready(pal['pal'])

    if DEBUG:
        im_low.save(output_path + base + '_lo.gif')
        im_high.save(output_path + base +'_hi.gif')

   
    def check_tile_duplicate(data,reference_db):
        for k,ref in enumerate(reference_db):
            if (np.all(data == ref)):
                return k
        return -1 #no result found

    def format_gb(im):
        full = np.frombuffer(im.tobytes(),dtype='uint8')
        full = full.reshape((im.height,im.width))

        map_data =np.array([],dtype='uint8')

        tile_data_list = []
        tiles_added = 0
        dups = 0
         #each tile 8x8 pixels: 64 pixel
        #2 bytes per row, i.e. 16 bits, 2 bit per pixel
        #first byte is all msb
        #second byte is all lsb
        for i in np.arange(int(im.height/8)):
            for j in np.arange(int(im.width/8)):
                px = j * 8 
                py = i * 8
                rows = np.array(np.arange(px,px+8),dtype=np.intp)
                columns = np.array(np.arange(py,py+8),dtype=np.intp)
                tile = full [np.ix_(columns,rows)] #careful here with row/column order
                
                tile_bytes = np.array([],dtype='uint8')
                for row in tile:
                    msb,lsb=0,0
                    for d in row.tobytes():
                        msb = msb << 1
                        lsb = lsb << 1
                        msb = msb | ((d & 2) >> 1)
                        lsb = lsb | (d & 1)
                    
                    tile_bytes = np.append(tile_bytes,lsb)#first is lsb
                    tile_bytes = np.append(tile_bytes,msb)#second is msb
                
                res = check_tile_duplicate(tile_bytes,tile_data_list)
                                
                if (res == -1) : #FIXME: incompatible with result finding in 8800 adressing mode (signed)
                    tile_data_list.append(tile_bytes)
                    map_data= np.append(map_data,tiles_added) #store the tile index in 8000 addressing mode
                    tiles_added +=1
                else:
                    dups +=1 
                    map_data = np.append(map_data,res)

        print('{:d} dups found for {:d} tiles '.format(dups,int(im.height* im.width / 64)))

        tile_data = np.asarray(tile_data_list).astype('uint8').flatten()
        return tile_data,map_data
    
    import binascii
    from textwrap import wrap

    
    template = '''
; from file: {file:s}
; Pixel Width: {width:d}px
; Pixel Height: {height:d}px


{name:s}_tile_map_size EQU ${map_size:02X}
{name:s}_tile_map_width EQU ${map_width:02X}
{name:s}_tile_map_height EQU ${map_height:02X}

{name:s}_tile_data_size EQU ${tile_data_size:02X}
{name:s}_tile_count EQU ${tile_count:02X}

{name:s}_palette_order_size EQU ${palette_data_size:02X}


{name:s}_palette_order:
DB {palette_data:s}

{name:s}_map_data: 
DB {map_data:s}

{name:s}_tile_data: 
DB {tile_data:s}
'''
    tile_len_total = 0
    tile_data, map_data = format_gb(im_low)
    
    tile_len_total += len(tile_data)
    
    # palette_data_string= pal_byteready.tobytes().hex()
    # palette_data_string= '$' + ',$'.join(wrap(palette_data_string,2))
    
    def format_hexstring(data):
        data_str = data.astype('uint8').tobytes().hex()
        data_str = '$' + ',$'.join(wrap(data_str,2))
        return data_str

    with open(output_path + base + '.inc','w') as f:
        info = {
        'palette_data': format_hexstring(pal_byteready),
        'palette_data_size': len(pal_byteready),
        'map_data': format_hexstring(map_data),
        'tile_data': format_hexstring(tile_data),
        'height': im.height,
        'width' : im.width,
         'map_width':  int(im.width/8),
         'map_height':  int(im.height/8),
         'map_size':    len(map_data),
         'tile_data_size': len(tile_data),
         'tile_count' :  int(len(tile_data)/16),
         'name': base+'_lo',
         'file': file
        }
        
        f.write(template.format(**info))
        
##Do the high bit in the same file, append to it
    tile_data, map_data = format_gb(im_high)
    tile_len_total += len(tile_data)

    with open(output_path + base +'.inc','a') as f:
        info = {
        'palette_data': format_hexstring(pal_byteready),
        'palette_data_size': len(pal_byteready),
        'map_data': format_hexstring(map_data),
        'tile_data': format_hexstring(tile_data),
        'height': im.height,
        'width' : im.width,
         'map_width':  int(im.width/8),
         'map_height':  int(im.height/8),
         'map_size':    len(map_data),
         'tile_data_size': len(tile_data),
         'tile_count' :  int(len(tile_data)/16),
         'name': base+'_hi',
         'file': file
        }
        
        f.write(template.format(**info))

    if ((tile_len_total) > 3*2048): print("Warning! tile data larger than 3*2048 bytes! This will never fit")


if __name__ == "__main__":
    app_name = 'Pixelfall v{version} - Game Boy TileMaker and sequencer by {author}.'.format(version=__version__, author=__author__)
    parser = argparse.ArgumentParser(description=app_name)
    parser.add_argument('file_path', help='Gif file to process')
    parser.add_argument('--palette-mode', help='Enable color cycling in 1..5 fashion or extend palette to solid 0..6', type=str, default='cycle', choices=['extend','cycle'])
    parser.add_argument('--output_path', help='Output path',type=str, default='./')
    parser.add_argument('--debug', help='Enable debug output', action='store_true')


    args = parser.parse_args()

    if (args.debug):
        DEBUG = True

    main(args.file_path,args.output_path,args.palette_mode)