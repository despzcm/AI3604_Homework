#!/usr/bin/env python3
import cv2
import numpy as np
import sys



def binarize(gray_image, thresh_val):
  # TODO: 255 if intensity >= thresh_val else 0
  binary_image = np.zeros(gray_image.shape)
  for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
      binary_image[i][j] = 255 if gray_image[i][j] >= thresh_val else 0
      
  return binary_image

def label(binary_image):
  label=1
  labeled_image = np.zeros(binary_image.shape)
  label_set={}
  global set_num
  set_num=0
  for i in range(binary_image.shape[0]):
    for j in range(binary_image.shape[1]):
      if binary_image[i][j]==255:
        if i==0 and j==0:
          labeled_image[i][j] = label
          label+=1
        elif i==0:
          if labeled_image[i][j-1]!=0:
            labeled_image[i][j] = labeled_image[i][j-1]
          else:
            labeled_image[i][j] = label
            label+=1
        elif j==0:
          if labeled_image[i-1][j]!=0:
            labeled_image[i][j] = labeled_image[i-1][j]
          else:
            labeled_image[i][j] = label
            label+=1
        else:
          if labeled_image[i-1][j]==0 and labeled_image[i][j-1]==0 and labeled_image[i-1][j-1]==0:
            labeled_image[i][j] = label
            label+=1
          elif labeled_image[i-1][j]!=0 and labeled_image[i][j-1]==0 and labeled_image[i-1][j-1]==0:
            labeled_image[i][j] = labeled_image[i-1][j]
          elif labeled_image[i][j-1]!=0 and labeled_image[i-1][j]==0 and labeled_image[i-1][j-1]==0:
            labeled_image[i][j] = labeled_image[i][j-1]
          elif labeled_image[i-1][j-1]!=0 and labeled_image[i-1][j]==0 and labeled_image[i][j-1]==0:
            labeled_image[i][j] = labeled_image[i-1][j-1]
          elif labeled_image[i-1][j]!=0 and labeled_image[i][j-1]!=0 and labeled_image[i-1][j-1]==0:
            if labeled_image[i-1][j]==labeled_image[i][j-1]:
              labeled_image[i][j] = labeled_image[i-1][j]
            else:
              labeled_image[i][j] = labeled_image[i-1][j]
              if label_set=={}:
                label_set[set_num]=[labeled_image[i-1][j],labeled_image[i][j-1]]
                set_num+=1
              else:
                find_set=False
                for s1 in range(set_num):
                  for s2 in range(set_num):
                    if labeled_image[i-1][j] in label_set[s1] and labeled_image[i][j-1] in label_set[s2] and s1!=s2:
                      max_s=max(s1,s2)
                      min_s=min(s1,s2)
                      label_set[min_s]+=label_set[max_s]
                      label_set[max_s]=label_set[set_num-1] if set_num-1!=max_s else []
                      set_num-=1
                      find_set=True
                      break
                if not find_set:
                  for set in label_set:
                    if labeled_image[i-1][j] in label_set[set]:
                      label_set[set].append(labeled_image[i][j-1])
                      find_set=True
                      break
                    if labeled_image[i][j-1] in label_set[set]:
                      label_set[set].append(labeled_image[i-1][j])
                      find_set=True
                      break
                  if not find_set:
                    label_set[set_num]=[labeled_image[i-1][j],labeled_image[i][j-1]]
                    set_num+=1
          else:
            labeled_image[i][j] = labeled_image[i-1][j-1]
                  
            
  if set_num>0:
    color=255//set_num
  else:
    color=255
  # print(label)
  # print(set_num)
  for i in range(binary_image.shape[0]):
    for j in range(binary_image.shape[1]):
      if binary_image[i][j]!=0:
        for k in range(set_num):
          if labeled_image[i][j] in label_set[k]:
            labeled_image[i][j]=color*(k+1)
            break

  return labeled_image

def cal_E(theta,a,b,c):
  return a*np.square(np.sin(theta))-b*np.sin(theta)*np.cos(theta)+c*np.square(np.cos(theta))

def get_attribute(labeled_image):
  # TODO
  attribute_list = [] 
  color=255//set_num
  for k in range(set_num):
    item_list=np.argwhere(labeled_image==color*(k+1))
    x=item_list[:,1]
    y=labeled_image.shape[0]-item_list[:,0]-1
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    x_=x-x_mean
    y_=y-y_mean
    a=np.sum(np.square(x_))
    b=2*np.sum(x_*y_)
    c=np.sum(np.square(y_))
    theta_1=0.5*np.arctan(b/(a-c))
    theta_2=theta_1+np.pi/2
    E_min=cal_E(theta_1,a,b,c)
    E_max=cal_E(theta_2,a,b,c)
    if E_min>E_max:
      E_min,E_max=E_max,E_min
      theta_1,theta_2=theta_2,theta_1
    roundness=E_min/E_max
    x_mean=float(x_mean)
    y_mean=float(y_mean)
    theta_1=float(theta_1)
    roundness=float(roundness)
    attribute={'x':x_mean,'y':y_mean,'orientation':theta_1,'roundness':roundness} 
    attribute_list.append(attribute)
  return attribute_list

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labeled_image = label(binary_image)
  attribute_list = get_attribute(labeled_image)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
  # print(set_num)
  print(attribute_list)


if __name__ == '__main__':
  main(sys.argv[1:])
