import cv2
import os
from datetime import datetime

# Load the images, this will become the ingredients to craft the sequence, the smaller image
# will be superimposed on the iteratively, generating multiple images with fixed steps
step = 200 # How big each step is
preset_offset = 0 # Positive number means moving down in the y_direction
show_image = True # Change to false if you don't want to show the sequence
img_large = cv2.imread('Sequence_Inputs/Base.jpg')
img_small = cv2.imread('Sequence_Inputs/Bar_6.jpg')
img_dir = 'Sequence_Outputs'  # Where to output images

def superimpose(img_large, img_small, yoffset):
    skip_frame = False
    # Get the dimensions of the smaller image
    height, width, channels = img_small.shape
    # Calculate the coordinates where to place the smaller image
    x = int((img_large.shape[1] - width) / 2) - 50
    y = int((img_large.shape[0] - height) / 2) + yoffset + preset_offset
    # Create a Region of Interest (ROI) in the larger image and assign the smaller image to it
    if y < 0 or y > 1944:
        skip_frame = True
    if (y + height) < 0 or (y + height) > 1944:
        skip_frame = True
    if not skip_frame:
        roi = img_large[y:y+height, x:x+width]
        # print(y+height)
        img_small_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_small_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img_large_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img_small_fg = cv2.bitwise_and(img_small, img_small, mask=mask)
        dst = cv2.add(img_large_bg, img_small_fg)
        img_large[y:y+height, x:x+width] = dst
    return img_large, skip_frame

def main():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    os.makedirs(os.path.join(img_dir,dt_string))
    j = 1
    for i in range(10, -10, -1):
        img_large_clone = img_large.copy()
        img, skip = superimpose(img_large_clone, img_small, i * step)
        if not skip:
            if show_image:
                img_large_clone_resized = cv2.resize(img_large_clone, (1000, 750))
                cv2.imshow('Sequence Image', img_large_clone_resized)
                cv2.waitKey(100)
                img_name = f'{j:02d}.jpg'
                img_path = os.path.join(img_dir, dt_string)
                img_path = os.path.join(img_path, img_name)
                cv2.imwrite(img_path, img)
                j += 1

if __name__ == "__main__":
    main()
