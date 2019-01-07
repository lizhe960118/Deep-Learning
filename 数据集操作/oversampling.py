import os
from PIL import Image
from PIL import ImageFilter

filelist = train['Image'].loc[(train['cnt_freq'] < 10)].tolist()

for count in range(0, 2):
    for imagefile in filelist:
        os.chdir('/home/paperspace/fastai/courses/dll/data/humpback/train')
        im = Image.open(imagefile)
        im = im.convert('RGB')
        r, g, b = im.split()
        r = r.convert("RGB")
        g = g.convert("RGB")
        b = b.convert("RGB")
        im_blur = im.filer(ImageFilter.GaussianBlur)
        im_unsharp = im.filer(ImageFilter.UnsharpMask)

        os.chdir('/home/paperspace/fastai/courses/dll/data/humpback/copy')
        r.save(str(count)+'r_'+imagefile)
        g.save(str(count)+'g_'+imagefile)
        b.save(str(count)+'b_'+imagefile)
        im_blur.save(str(count) + 'bl_' + imagefile)
        im_unsharp.save(str(count) + 'un_' + imagefile)
