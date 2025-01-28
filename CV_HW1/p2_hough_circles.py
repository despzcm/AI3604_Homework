#!/usr/bin/env python3
import cv2
import numpy as np
import sys

def detect_edges(image):
    """Find edge points in a grayscale image.

    Args:
    - image (2D uint8 array): A grayscale image.

    Return:
    - edge_image (2D float array): A heat map where the intensity at each point
        is proportional to the edge magnitude.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    edge_image = np.zeros(image.shape)
    kernel_size = sobel_x.shape[0]
    image_padded = np.pad(image, (kernel_size//2, kernel_size//2), 'constant')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            G_x=np.sum(np.multiply(sobel_x,image_padded[i:i+kernel_size,j:j+kernel_size]))
            G_y=np.sum(np.multiply(sobel_y,image_padded[i:i+kernel_size,j:j+kernel_size]))
            G=np.sqrt(G_x**2+G_y**2)
            edge_image[i][j]=G
    edge_image=edge_image/np.max(edge_image)*255
    return edge_image
    raise NotImplementedError  #TODO


def hough_circles(edge_image, edge_thresh, radius_values):
    """Threshold edge image and calculate the Hough transform accumulator array.

    Args:
    - edge_image (2D float array): An H x W heat map where the intensity at each
        point is proportional to the edge magnitude.
    - edge_thresh (float): A threshold on the edge magnitude values.
    - radius_values (1D int array): An array of R possible radius values.

    Return:
    - thresh_edge_image (2D bool array): Thresholded edge image indicating
        whether each pixel is an edge point or not.
    - accum_array (3D int array): Hough transform accumulator array. Should have
        shape R x H x W.
    """
    edge_thresh=255*edge_thresh
    
    thresh_edge_image = np.zeros(edge_image.shape)
    for i in range(edge_image.shape[0]):
        for j in range(edge_image.shape[1]):
            if float(edge_image[i][j])>=edge_thresh:
                thresh_edge_image[i][j]=255
            else:
                thresh_edge_image[i][j]=0
    
    accum_array = np.zeros((len(radius_values),edge_image.shape[0],edge_image.shape[1]))
    # print(accum_array)
    # print(accum_array.shape)
    # print(type(accum_array[1][0][0]))
    theta = np.radians(np.arange(360))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for r in range(len(radius_values)):
        for i in range(edge_image.shape[0]):
            for j in range(edge_image.shape[1]):
                if thresh_edge_image[i][j]==255:
                    for k in range(360):
                        a = int(i - radius_values[r] * cos_theta[k])
                        b = int(j - radius_values[r] * sin_theta[k])
                        if a >= 0 and a < edge_image.shape[0] and b >= 0 and b < edge_image.shape[1]:
                            accum_array[r][a][b] += 1
    # print(accum_array[1][0][0])
    # print(type(accum_array))
    # print(accum_array.shape)
    return thresh_edge_image, accum_array
    raise NotImplementedError  #TODO


def find_circles(image, accum_array, radius_values, hough_thresh):
    """Find circles in an image using output from Hough transform.

    Args:
    - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
        original color image instead of its grayscale version so the circles
        can be drawn in color.
    - accum_array (3D int array): Hough transform accumulator array having shape
        R x H x W.
    - radius_values (1D int array): An array of R radius values.
    - hough_thresh (int): A threshold of votes in the accumulator array.

    Return:
    - circles (list of 3-tuples): A list of circle parameters. Each element
        (r, y, x) represents the radius and the center coordinates of a circle
        found by the program.
    - circle_image (3D uint8 array): A copy of the original image with detected
        circles drawn in color.
    """
    circle_list=[]
    # print(accum_array)
    
    for r in range(len(radius_values)):
        for i in range(accum_array.shape[1]):
            for j in range(accum_array.shape[2]):
                if accum_array[r][i][j]>=hough_thresh:
                    r1=max(0,r-2)
                    r2=min(len(radius_values)-1,r+3)
                    if accum_array[r][i][j]==np.max(accum_array[r1:r2+1,i-2:i+4,j-2:j+4]):
                        circle_list.append((radius_values[r],i,j))
                    
    circle_image = image.copy()
    for circle in circle_list:
        cv2.circle(circle_image, (circle[2], circle[1]), circle[0], (0, 255, 0), 2)
    return circle_list, circle_image
    raise NotImplementedError  #TODO


def main(argv):
    img_name = argv[0]
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_image = detect_edges(gray_image)
    edge_thresh = float(argv[1])
    hough_thresh = int(argv[2])
    radius_values = [r for r in range(20,41)]
    thresh_edge_image,accum_array= hough_circles(edge_image, edge_thresh, radius_values)   
    circle_list,circle_image=find_circles(img,accum_array,radius_values,hough_thresh)
    cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
    cv2.imwrite('output/' + img_name + "_edge_detect.png", edge_image)
    cv2.imwrite('output/' + img_name + "_edge.png", thresh_edge_image)
    print(circle_list)
    print(len(circle_list)) 
    cv2.imwrite('output/' + img_name + "_circle.png", circle_image)





if __name__ == '__main__':
  #TODO
    main(sys.argv[1:])
