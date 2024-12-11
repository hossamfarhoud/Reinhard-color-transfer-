import numpy as np
import cv2
import os

# Change working directory
os.chdir(r"C:\Users\M&G\Downloads\color transfer")

input_dir = r"C:\Users\M&G\Downloads\color transfer\input_images\\"
output_dir = r"C:\Users\M&G\Downloads\color transfer\output_dir\\"
template_path = r"C:\Users\M&G\Downloads\color transfer\template_images\428690.jpg"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std

# Read and process the template image
template_img = cv2.imread(template_path)
if template_img is None:
    print("Error: Template image not found.")
    exit()

template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB)
template_mean, template_std = get_mean_and_std(template_img)

# Process input images
input_image_list = os.listdir(input_dir)
for img in input_image_list:
    print(f"Processing: {img}")
    input_img_path = os.path.join(input_dir, img)
    input_img = cv2.imread(input_img_path)
    if input_img is None:
        print(f"Error: Unable to read image {input_img_path}")
        continue

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    img_mean, img_std = get_mean_and_std(input_img)

    # Apply color transfer using NumPy for optimization
    input_img = ((input_img - img_mean) * (template_std / img_std)) + template_mean
    input_img = np.clip(input_img, 0, 255)  # Ensure values are in valid range
    input_img = np.round(input_img).astype(np.uint8)  # Convert to uint8

    # Convert back to BGR and save the image
    input_img = cv2.cvtColor(input_img, cv2.COLOR_LAB2BGR)
    output_img_path = os.path.join(output_dir, f"modified_{img}")
    cv2.imwrite(output_img_path, input_img)

print("Color transfer completed!")
