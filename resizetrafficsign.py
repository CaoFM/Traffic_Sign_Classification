import numpy as np
from PIL import Image
import glob

files = glob.glob("my_img/*.jpg")
print(files[0])
for fname in files:
    image = Image.open(fname)
    new_image = image.resize((32, 32))
    #new_image.show()
    #print(type(new_image))
    #new_image.save('%s_sized.jpg' % fname)

#    np_img = np.array(new_image)
#    print(np_img.shape)
#    print(type(np_img))
