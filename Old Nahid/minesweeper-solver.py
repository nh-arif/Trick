import cv2
import numpy as np
import os

def extract_and_resize_shapes(image_path, output_folder, target_size=(23, 23)):
    # Load the source image with boxes
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    unique_shapes = []
    unique_contours = []

    # Identify unique shapes
    for cnt in contours:
        # Approximate the contour to reduce points
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if this contour is similar to any existing unique contour
        is_unique = True
        for unique in unique_shapes:
            similarity = cv2.matchShapes(approx, unique, cv2.CONTOURS_MATCH_I1, 0.0)
            if similarity < 0.1:  # Adjust threshold as necessary
                is_unique = False
                break
        
        if is_unique:
            unique_shapes.append(approx)
            unique_contours.append(cnt)

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save each unique shape as a resized PNG image
    for i, cnt in enumerate(unique_contours):
        # Create a mask for the current shape
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Bitwise the original image and mask to isolate the shape
        isolated_shape = cv2.bitwise_and(image, mask)

        # Resize the isolated shape to the target size (23x23)
        resized_shape = cv2.resize(isolated_shape, target_size, interpolation=cv2.INTER_AREA)

        # Convert the image to grayscale and apply a binary threshold to save as PNG
        resized_gray = cv2.cvtColor(resized_shape, cv2.COLOR_BGR2GRAY)
        _, resized_thresh = cv2.threshold(resized_gray, 1, 255, cv2.THRESH_BINARY)

        # Save the unique shape as a PNG file
        output_filename = f"{output_folder}/shape_{i}.png"
        cv2.imwrite(output_filename, resized_thresh)

# Example usage
extract_and_resize_shapes('./ByBit-mine-sweeper/box.png', './ByBit-mine-sweeper/shapes')
