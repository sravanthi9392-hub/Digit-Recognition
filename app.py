import cv2
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import os

os.system('cls' if os.name == 'nt' else 'clear')

# =====================================================================
# 1. TRAIN HIGH-ACCURACY NEURAL NETWORK MODEL (MNIST 28x28)
# =====================================================================
print("=" * 70)
print(" LOADING HIGH-RESOLUTION MNIST DATASET (28x28 Pixels)")
print("=" * 70)

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target
X_train, y_train = X[:20000], y[:20000]

print("\nTraining an Artificial Neural Network Classifier...")
classifier = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30, alpha=1e-4,
                           solver='adam', random_state=1)
classifier.fit(X_train, y_train)
print("✨ Advanced Neural Network trained successfully!")
print("=" * 70)

# =====================================================================
# 2. DRAWING & HANDLER SETUP
# =====================================================================
canvas = np.zeros((500, 500), dtype=np.uint8)
drawing = False 
prev_x, prev_y = -1, -1
predicted_digit = "None"

def draw_line(event, x, y, flags, param):
    global drawing, prev_x, prev_y, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev_x, prev_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Stroke thickness optimized perfectly for 28x28 grids
            cv2.line(canvas, (prev_x, prev_y), (x, y), 255, 28)
            prev_x, prev_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Touchpad Digit Canvas")
cv2.setMouseCallback("Touchpad Digit Canvas", draw_line)

print("\n--- Easy Controls ---")
print("-> Click and DRAG on the window to DRAW a digit")
print("-> Press 'p' to PREDICT the digit using the Neural Network")
print("-> Press 'c' to CLEAR the window completely")
print("-> Press 'q' to QUIT the program")

while True:
    display_frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    display_frame[canvas == 255] = [0, 255, 0] # Bright Green Ink

    cv2.putText(display_frame, f"AI Guess: {predicted_digit}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    cv2.imshow("Touchpad Digit Canvas", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((500, 500), dtype=np.uint8)
        predicted_digit = "None"
    elif key == ord('p'):
        pts = np.argwhere(canvas == 255)
        if len(pts) > 0:
            y_min, x_min = pts.min(axis=0)
            y_max, x_max = pts.max(axis=0)
            
            # Crop drawn area out dynamically
            crop = canvas[y_min:y_max+1, x_min:x_max+1]
            
            if crop.size > 0:
                # 1. Resize the core drawing down to a standard 20x20 box inside the image
                cy_h, cx_w = crop.shape
                if cy_h > cx_w:
                    new_h = 20
                    new_w = int(round(cx_w * (20.0 / cy_h)))
                    if new_w < 1: new_w = 1
                else:
                    new_w = 20
                    new_h = int(round(cy_h * (20.0 / cx_w)))
                    if new_h < 1: new_h = 1
                    
                resized_digit = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 2. Place the digit inside a standard blank 28x28 background image array
                mnist_box = np.zeros((28, 28), dtype=np.uint8)
                
                # Temp position placement layout
                y_off = (28 - new_h) // 2
                x_off = (28 - new_w) // 2
                mnist_box[y_off:y_off+new_h, x_off:x_off+new_w] = resized_digit
                
                # 3. --- DYNAMIC CENTER-OF-MASS SHIFT ENGINE ---
                # Calculate pixel weights to locate the true spatial center
                M = cv2.moments(mnist_box)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    
                    # Shift distance calculation to align with center coordinate (14, 14)
                    shift_x = int(round(14 - cx))
                    shift_y = int(round(14 - cy))
                    
                    # Physically slide array elements into perfect center of gravity alignment
                    T = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    final_matrix = cv2.warpAffine(mnist_box, T, (28, 28))
                else:
                    final_matrix = mnist_box
                
                # Flatten matrix array row vectors
                feat_vector = final_matrix.reshape(1, -1)
                
                # Run updated class prediction 
                predicted_digit = str(classifier.predict(feat_vector)[0])
        else:
            predicted_digit = "Empty Canvas"

cv2.destroyAllWindows()