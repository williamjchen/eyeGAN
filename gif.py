import os
import imageio
import glob

# png_dir = 'training_images/'
# gif_images = []
# for file_name in os.listdir(png_dir):
#     if file_name.endswith('.png'):
#         file_path = os.path.join(png_dir, file_name)
#         gif_images.append(imageio.imread(file_path))
# imageio.mimsave(os.path.join(png_dir, 'dcgan.gif'), gif_images, fps=15)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('training_images/image*.png')
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
