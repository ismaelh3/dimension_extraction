import numpy as np
import cv2
import glob
import os
import pickle

# Camera calibration parameters
# You can modify these variables as needed
CHESSBOARD_SIZE = (8, 6)  # Number of inner corners per chessboard row and column
SQUARE_SIZE = 2.5          # Size of a square in centimeters (25mm from PDF)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Folder this script lives in, so paths work from any cwd
CALIBRATION_IMAGES_PATH = os.path.join(SCRIPT_DIR, 'calibration_images', '*.jpeg')  # Path to calibration images
OUTPUT_DIRECTORY = os.path.join(SCRIPT_DIR, 'output')  # Directory to save calibration results
SAVE_UNDISTORTED = True   # Whether to save undistorted images

def calibrate_camera():
    """
    Calibrate the camera using chessboard images.
    
    Returns:
        ret: The RMS re-projection error
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    
    # Scale object points by square size (for real-world measurements)
    objp = objp * SQUARE_SIZE
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    image_names = []  # Filenames corresponding to each entry in objpoints/imgpoints
    
    # Get list of calibration images
    images = glob.glob(CALIBRATION_IMAGES_PATH)
    
    if not images:
        print(f"No calibration images found at {CALIBRATION_IMAGES_PATH}")
        return None, None, None, None, None
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    print(f"Found {len(images)} calibration images")
    
    # Process each calibration image
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            image_names.append(fname)

            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            
            # Save image with corners drawn
            output_img_path = os.path.join(OUTPUT_DIRECTORY, f'corners_{os.path.basename(fname)}')
            cv2.imwrite(output_img_path, img)
            
            print(f"Processed image {idx+1}/{len(images)}: {fname} - Chessboard found")
        else:
            print(f"Processed image {idx+1}/{len(images)}: {fname} - Chessboard NOT found")
    
    if not objpoints:
        print("No chessboard patterns were detected in any images.")
        return None, None, None, None, None
    
    print("Calibrating camera...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None
)

    # Per-image error breakdown, so bad images can be spotted without a full reshoot
    per_image_errors = calculate_reprojection_errors(objpoints, imgpoints, mtx, dist, rvecs, tvecs, image_names)
    print_reprojection_report(per_image_errors, ret, OUTPUT_DIRECTORY)

    # Save calibration results
    calibration_data = {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        # Resolution the matrix is valid at (w, h). fx/fy/cx/cy are in PIXELS of
        # this image size — frames captured at any other resolution must have the
        # matrix rescaled before use (segmentation.py does this per frame).
        'image_size_wh': tuple(gray.shape[::-1]),
        'rotation_vectors': rvecs,
        'translation_vectors': tvecs,
        'reprojection_error': ret,
        'per_image_errors': per_image_errors
    }

    with open(os.path.join(OUTPUT_DIRECTORY, 'calibration_data.pkl'), 'wb') as f:
        pickle.dump(calibration_data, f)

    # Save camera matrix and distortion coefficients as text files
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'camera_matrix.txt'), mtx)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'distortion_coefficients.txt'), dist)

    print(f"Calibration complete! RMS re-projection error: {ret}")
    print(f"Results saved to {OUTPUT_DIRECTORY}")

    return ret, mtx, dist, rvecs, tvecs

def undistort_images(mtx, dist):
    """
    Undistort all calibration images using the calibration results.
    
    Args:
        mtx: Camera matrix
        dist: Distortion coefficients
    """
    if not SAVE_UNDISTORTED:
        return
    
    images = glob.glob(CALIBRATION_IMAGES_PATH)
    
    if not images:
        print(f"No images found at {CALIBRATION_IMAGES_PATH}")
        return
    
    undistorted_dir = os.path.join(OUTPUT_DIRECTORY, 'undistorted')
    if not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)
    
    print(f"Undistorting {len(images)} images...")
    
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        
        # Refine camera matrix based on free scaling parameter
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # Undistort image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Crop the image (optional)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # Save undistorted image
        output_img_path = os.path.join(undistorted_dir, f'undistorted_{os.path.basename(fname)}')
        cv2.imwrite(output_img_path, dst)
        
        print(f"Undistorted image {idx+1}/{len(images)}: {fname}")
    
    print(f"Undistorted images saved to {undistorted_dir}")

def calculate_reprojection_errors(objpoints, imgpoints, mtx, dist, rvecs, tvecs, image_names):
    """
    Calculate the RMS reprojection error for each calibration image individually.

    The per-image error is computed the same way as OpenCV's overall RMS
    (sqrt of the mean squared pixel distance), so it's directly comparable
    to the total reprojection error printed after calibration.

    Returns:
        List of (image_name, rms_error_px) tuples, sorted worst-first.
    """
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        diff = imgpoints[i].reshape(-1, 2) - imgpoints2.reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        errors.append((image_names[i], rms))

    errors.sort(key=lambda item: item[1], reverse=True)
    return errors

def print_reprojection_report(per_image_errors, overall_rms, output_directory, flag_multiple=1.5):
    """
    Print a worst-first per-image error report and flag images that are
    dragging the overall RMS up, so bad images can be spotted without
    reshooting the whole set. Also saves the report to a text file.
    """
    rms_values = [error for _, error in per_image_errors]
    median_error = float(np.median(rms_values))
    threshold = median_error * flag_multiple

    lines = [f"Overall RMS reprojection error: {overall_rms:.3f} px",
             f"Median per-image error: {median_error:.3f} px", ""]
    for name, error in per_image_errors:
        flag = "  <-- consider reviewing/retaking" if error > threshold else ""
        lines.append(f"{error:6.3f} px  {os.path.basename(name)}{flag}")

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(output_directory, 'reprojection_errors.txt'), 'w') as f:
        f.write(report + "\n")

def main():
    """
    Main function to run the camera calibration process.
    """
    print("Starting camera calibration...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    
    if mtx is None:
        print("Calibration failed. Exiting.")
        return
    
    # Undistort images
    undistort_images(mtx, dist)
    
    print("Camera calibration completed successfully!")

if __name__ == "__main__":
    main()