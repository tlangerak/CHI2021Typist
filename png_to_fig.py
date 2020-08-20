import imageio
import os


dir = "ABC_Results/images/"
filenames = []
for file in os.listdir(dir):
    if file.endswith(".png"):
        print(os.path.join(dir, file))
        filenames.append(os.path.join(dir, file))


# print(filenames)
# with imageio.get_writer('ABC_Results/images/combined.gif', mode='I') as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)

filenames.sort() # this iteration technique has no built in order, so sort the frames
images = list(map(lambda filename: imageio.imread(filename), filenames))
imageio.mimsave(os.path.join('my_very_own_gif.gif'), images, duration = 0.5) # modify the frame duration as needed
