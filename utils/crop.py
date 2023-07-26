
from PIL import Image


def crop_image(input_image, pixels_from_bottom):
                
                cropped_image = input_image[:,:]
                cropped_image = cropped_image[:-pixels_from_bottom, :]
                return cropped_image