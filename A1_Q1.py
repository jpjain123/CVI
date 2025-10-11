# Jayneel Pratap Jain
# 101983237
# jpjain@myseneca.ca
# ASSIGNMENT-1 - Part II: Invisible Cloak (GREEN, dark-green tuned)

import cv2
import numpy as np

def apply_invisible_cloak(frame, background):
    """
    Implements the invisible cloak logic for GREEN cloth.

    Steps (as required by the assignment):
    1) Convert the image to HSV.
    2) Create a mask for GREEN using cv2.inRange() (two ranges to cover variations).
    3) Use cv2.morphologyEx() on the mask (Opening removes small white noise specks;
       Closing fills tiny black holes). This cleans the mask.
    4) Segment out the green area using the cleaned mask.
    5) Replace the green area with the corresponding pixels from the background.
    """
    # Ensure background matches frame size (safety)
    h, w = frame.shape[:2]
    if background.shape[:2] != (h, w):
        background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1) BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

 
    lower_green_1 = np.array([35,  60,  25], dtype=np.uint8)  # allow darker green as I only had dark green mostly 
    upper_green_1 = np.array([55, 255, 255], dtype=np.uint8)
    lower_green_2 = np.array([56,  60,  25], dtype=np.uint8)
    upper_green_2 = np.array([85, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_green_1, upper_green_1)
    mask2 = cv2.inRange(hsv, lower_green_2, upper_green_2)
    mask = cv2.bitwise_or(mask1, mask2)

   
    S = hsv[:, :, 1]
    sat_guard = cv2.inRange(S, 70, 255)    
    mask = cv2.bitwise_and(mask, sat_guard)

    # 3) Clean mask using morphologyEx (with a 5x5 kernel)
    # morphologyEx applies morphological operations with a structuring element ("kernel").
    # - Opening  = Erode then Dilate  -> removes small white specks (noise) from the mask.
    # - Closing  = Dilate then Erode  -> fills tiny black holes inside the masked regions.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


    mask = cv2.dilate(mask, kernel, iterations=1)

    
    mask_inv = cv2.bitwise_not(mask)
    foreground_part = cv2.bitwise_and(frame, frame, mask=mask_inv)

 
    background_part = cv2.bitwise_and(background, background, mask=mask)
    final_output = cv2.add(foreground_part, background_part)

    return final_output

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    cv2.namedWindow("Background (press 'b' to lock)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Cloak Effect (q to quit)", cv2.WINDOW_NORMAL)

    background = None
    locked = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

      
        frame = cv2.flip(frame, 1)

        if not locked:
         
            cv2.imshow("Background (press 'b' to lock)", frame)
            cv2.imshow("Cloak Effect (q to quit)", frame)
        else:
           
            cv2.imshow("Background (press 'b' to lock)", background)
            output = apply_invisible_cloak(frame, background)
            cv2.imshow("Cloak Effect (q to quit)", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            background = frame.copy()
            locked = True
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
