import cv2
import os

cap = cv2.VideoCapture(0)  # Use the default camera (you may need to change the index based on your setup)
shirtFolderPath = "ilo/i.png"
listShirts = os.listdir(shirtFolderPath)
fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 581 / 440
imageNumber = 0
imgButtonRight = cv2.imread("f1.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

while True:
    success, img = cap.read()

    # Convert the frame to HSV for color-based hand detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of color for detecting the hand
    lower_skin = (0, 20, 70)
    upper_skin = (20, 255, 255)

    # Create a binary mask for the hand color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours in the mask to detect the hand region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Assuming the largest contour is the hand
        hand_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the hand
        x, y, w, h = cv2.boundingRect(hand_contour)

        # Calculate wrist landmarks based on the bounding box
        lm11 = (x, y + h // 2)
        lm12 = (x + w, y + h // 2)

        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

        # Resize shirt image based on wrist landmarks
        new_w = int((lm12[0] - lm11[0]) * fixedRatio)
        imgShirt = cv2.resize(imgShirt, (new_w, int(new_w * shirtRatioHeightWidth)))

        # Calculate position to overlay shirt
        x_offset = int(lm11[0] - (new_w - (lm12[0] - lm11[0])) / 2)
        y_offset = int(lm11[1])

        # Create a mask for the shirt
        mask = imgShirt[:, :, 3]

        # Overlay shirt on the frame
        img[y_offset:y_offset + imgShirt.shape[0], x_offset:x_offset + imgShirt.shape[1]] = \
            img[y_offset:y_offset + imgShirt.shape[0], x_offset:x_offset + imgShirt.shape[1]] * \
            (1 - mask[:, :, None] / 255.0) + imgShirt[:, :, :3] * (mask[:, :, None] / 255.0)

        # Draw buttons on the frame
        img[10:10 + imgButtonRight.shape[0], 10:10 + imgButtonRight.shape[1]] = imgButtonRight
        img[10:10 + imgButtonLeft.shape[0], img.shape[1] - 10 - imgButtonLeft.shape[1]:img.shape[1] - 10] = imgButtonLeft

    cv2.imshow("Live Camera Try-On", img)

    key = cv2.waitKey(1)  # Wait for a key event
    if key == 27:  # Exit when 'Esc' key is pressed
        break

    # Handle button clicks
    if 10 <= lm11[0] <= 10 + imgButtonRight.shape[1] and 10 <= lm11[1] <= 10 + imgButtonRight.shape[0]:
        counterRight += 1
        if counterRight * selectionSpeed > 360:
            counterRight = 0
            if imageNumber < len(listShirts) - 1:
                imageNumber += 1

    elif img.shape[1] - 10 - imgButtonLeft.shape[1] <= lm11[0] <= img.shape[1] - 10 and \
            10 <= lm11[1] <= 10 + imgButtonLeft.shape[0]:
        counterLeft += 1
        if counterLeft * selectionSpeed > 360:
            counterLeft = 0
            if imageNumber > 0:
                imageNumber -= 1

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()