import os
import imageio
import glob

anim_file = 'dcgan_c_p4.gif'

with imageio.get_writer(anim_file, mode='I', fps=25) as writer:
    filenames = glob.glob('training_images_c_3_p4/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

