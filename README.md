﻿# AI3604_Homework

## HW1
### **Written Assignment**  
1. **Pinhole Camera & Perspective Projection**  
   - **a.** Determine the shape of a circular disk (parallel to the image plane) under perspective projection. Derive equations to show it remains a circle.  
   - **b.** Analyze vanishing points for lines in specific planes (e.g., \(B=1\) and \(A=1\) cases). Compute vanishing points for three line directions in each plane.  
   - **c.** Generalize the relationship between plane parameters (\(A, B, C, D\)) and vanishing point locations.  

---

### **Programming Assignment**  
#### **Problem 1: Object Characterization**  
**Goal:** Detect 2D objects, compute their positions, orientations, and roundedness.  
- **a. Binarization:** Convert a grayscale image to binary using a threshold. Output: 255 if pixel ≥ threshold, else 0.  
- **b. Sequential Labeling:** Implement a two-pass or recursive algorithm to label connected regions. Handle label equivalences.  
- **c. Object Attributes:** For each labeled region, compute:  
  - **Position:** Centroid (origin at bottom-left).  
  - **Orientation:** Principal axis angle (in radians, using moments/PCA).  
  - **Roundedness:** Compare area to perimeter or fit to an ideal circle.  

**Coordinate System:** Origin at bottom-left for Problem 1.  

---

#### **Problem 2: Hough Transform for Circles**  
**Goal:** Detect circles in an image (e.g., `coins.png`).  
- **a. Edge Detection:** Implement Sobel operators to compute edge magnitudes. Output: Edge magnitude map.  
- **b. Hough Transform:** Build a 3D accumulator array for circle parameters \((x, y, r)\). Threshold edge magnitudes to retain strong edges.  
- **c. Circle Detection:** Extract high-vote candidates from the accumulator. Draw detected circles on the original image.  

**Key Parameters:**  
- Edge threshold (`edge_thresh`) to filter weak edges.  
- Hough vote threshold (`hough_thresh`) to identify valid circles.  

**Coordinate System:** Origin at top-left for Problems 2 and 3.  

---


## HW2

### **Programming Assignment (85 points)**  
1. **Objective**: Perform camera calibration using chessboard images to determine intrinsic parameters (focal length, principal point) and compute projection errors.  
2. **Tools**: Basic Python, NumPy, and limited OpenCV functions (e.g., `cv2.imread`, `cv2.findChessboardCorners`). Avoid using high-level "magic" functions for direct solutions.  
3. **Steps**:  
   - Define 3D world coordinates of chessboard corners (simplified as grid points, e.g., `(0,0)`, `(1,0)`, etc.).  
   - Detect pixel coordinates `(u, v)` of corners in multiple images.  
   - Use linear algebra to solve for camera parameters and 3D-to-2D correspondences.  
4. **Outputs**:  
   - Intrinsic parameters (focal length, principal point).  
   - Projection error per image (reprojection error).  
5. **Notes**:  
   - Use central image regions to minimize distortion impact.  
   - Results are scaled to chessboard square size.  

### **Written Assignment (15 points)**  
1. **Problem 1**:  
   - **Part (a)**: Derive the optimal transformation `(A*, T*)` minimizing the squared error sum for noisy 3D point correspondences.  
     - Key steps: Use centroid alignment `(X̂, Ŷ)` and solve via least squares.  
   - **Part (b)**: Prove that **4 correspondences** are the minimum required to estimate the 3D transformation `(A, T)`.  
     - Reasoning: Degrees of freedom analysis (3 rotation, 3 translation, scaling).  




