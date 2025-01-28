import cv2
import numpy as np
import glob

def read_images(image_directory):
    # Read all jpg images from the specified directory
    return [cv2.imread(image_path) for image_path in glob.glob(f"{image_directory}/*.jpg")]

def find_image_points(images, pattern_size):
    world_points = []
    image_points = []
    
    # TODO: Initialize the chessboard world coordinate points
    def init_world_points(pattern_size):
        # Students should fill in code here to generate the world coordinates of the chessboard
        w=pattern_size[0]
        h=pattern_size[1]
        x = np.arange(0, w)
        y = np.arange(0, h)
        xx, yy = np.meshgrid(x, y)
        coords = np.vstack((xx.ravel(), yy.ravel())).T.astype(np.float32)
        return coords
        
    
    # TODO: Detect chessboard corners in each image
    def detect_corners(image, pattern_size):
        # Students should fill in code here to detect corners using cv2.findChessboardCorners or another method
        gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, pattern_size, None)
        if ret: 
            return corners.reshape(-1, 2)
        else:
            return None

    # TODO: Complete the loop below to obtain the corners of each image and the corresponding world coordinate points
    for image in images:
        corners = detect_corners(image, pattern_size)
        if corners is not None:
            # Add image corners
            image_points.append(corners)
            # Add the corresponding world points
            world_points.append(init_world_points(pattern_size))
    
    return world_points, image_points

def calibrate_camera(world_points, image_points):
    assert len(world_points) == len(image_points), "The number of world coordinates and image coordinates must match"
    
    num_points = len(world_points)
    A = []
    B = []
    K = np.zeros((4, 4))
    P = None

    # TODO main loop, use least squares to solve for P and then decompose P to get K and R
    # The steps are as follows:
    # 1. Construct the matrix A and B
    # 2. Solve for P using least squares
    # 3. Decompose P to get K and R
    V=[]
    P=[]
    
    for i in range(num_points):
        A=[]
        # print(image_points[i].shape)
        image_point=image_points[i]
        world_point=world_points[i]
        index = []
        for j in range(world_point.shape[0]):
            x,y=world_point[j]
            if x>=8 and x<=24 and y>=4 and y<=20:
                index.append(j)
        world_point, image_point = world_point[index], image_point[index]
        for j in range(world_point.shape[0]):   
            x, y = image_point[j]
            X, Y = world_point[j]
            A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y,-x])
            A.append([0, 0, 0, X, Y, 1, -y * X, -y * Y,-y])

        
       
        A = np.array(A)
        # print(A.shape)
        # print(A)
        eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
        min_eigenvalue_index = np.argmin(eigenvalues)
        H= eigenvectors[:, min_eigenvalue_index].reshape(3, 3)
        P.append(H)
        h0=H[:,0]
        h1=H[:,1]
        h2=H[:,2]
        v01=[h0[0]*h1[0], h0[0]*h1[1]+h0[1]*h1[0], h0[1]*h1[1], h0[2]*h1[0]+h0[0]*h1[2], h0[2]*h1[1]+h0[1]*h1[2], h0[2]*h1[2]]
        v00=[h0[0]*h0[0], h0[0]*h0[1]+h0[1]*h0[0], h0[1]*h0[1], h0[2]*h0[0]+h0[0]*h0[2], h0[2]*h0[1]+h0[1]*h0[2], h0[2]*h0[2]]
        v11=[h1[0]*h1[0], h1[0]*h1[1]+h1[1]*h1[0], h1[1]*h1[1], h1[2]*h1[0]+h1[0]*h1[2], h1[2]*h1[1]+h1[1]*h1[2], h1[2]*h1[2]]
        V.append(v01)
        V.append(np.array(v00)-np.array(v11))
        
        
    V=np.array(V)
    eigenvalues, eigenvectors = np.linalg.eig(V.T @ V)
    min_eigenvalue_index = np.argmin(eigenvalues)
    b= eigenvectors[:, min_eigenvalue_index]
    #symmetric matrix
    
    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])
    
    L = np.linalg.cholesky(B)
    K = np.linalg.inv(L.T)
    
    normlize=K[2,2]
    K=K/normlize

    # Please ensure that the diagonal elements of K are positive
    
    return K, P

# Main process
image_path = 'Sample_Calibration_Images'
images = read_images(image_path)

# TODO: I'm too lazy to count the number of chessboard squares, count them yourself
pattern_size = (31, 23)  # The pattern size of the chessboard ,the number of inner corners per a chessboard row and column
world_points, image_points = find_image_points(images, pattern_size)

camera_matrix, camera_extrinsics = calibrate_camera(world_points, image_points)

print("Camera Calibration Matrix:")
print(camera_matrix)
# print("Camera Extrinsics:")
# print(camera_extrinsics)

def test(image_directory, pattern_size):
    # In this function, you are allowed to use OpenCV to verify your results. This function is optional and will not be graded.
    # return None, directly print the results
    # TODO
    images = read_images(image_directory)
    world_points, image_points = find_image_points(images, pattern_size)
    image_size = images[0].shape[:2][::-1]
    
    expand_vector = np.zeros((world_points[0].shape[0], 1), dtype=np.float32)
    object_points=[np.append(view, expand_vector, axis=1) for view in world_points]
    _, camera_matrix, _, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    print("Camera Calibration Matrix by OpenCV:")
    print("Camera Matrix:\n", camera_matrix)

def reprojection_error(world_points, image_points, camera_extrinsics):
    # In this function, you are allowed to use OpenCV to verify your results.
    # show the reprojection error of each image
    errorOFimages=[]
    for i in range(len(world_points)):
    
        object_points = np.hstack([world_points[i], np.ones((world_points[i].shape[0], 1), dtype=np.float32)])

       
        projected_points = camera_extrinsics[i] @ object_points.T 

        projected_points /= projected_points[2, :]  # Normalize the projected points

        norm_error = np.sqrt(np.sum((image_points[i].T - projected_points[:2, :])**2, axis=0))
        errorOFimages.append(np.mean(norm_error))

    print("Reprojection Error:")
    print(errorOFimages)
    
    
    
test(image_path, pattern_size)
reprojection_error(world_points, image_points, camera_extrinsics)