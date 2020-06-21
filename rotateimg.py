import numpy as np
from PIL import Image

image = Image.open("my_img/stop_sized.jpg")


new_image = image.rotate(-30)

new_image.show()
print(type(new_image))


new_arr = np.array(new_image)
print(type(new_arr))
print(new_arr.shape)
#new_image.save('%s_sized.jpg' % fname)
