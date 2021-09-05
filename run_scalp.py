# Jacob Miske
# GPL License
#!/usr/bin/python3

from operator import sub
import os, sys, random
from matplotlib import image
import numpy as np
from numpy.core.fromnumeric import size
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import cv2
import cmd
from pyfiglet import Figlet

class ScALP(cmd.Cmd):
  custom_fig = Figlet(font='slant')
  intro = 'Welcome to the ScALP CLI for Raspberry Pi \n'
  prompt = '> '
  file = None
  print(custom_fig.renderText(' ScALP '))
  
  def do_background(self, arg):
    """
    Generate background video
    """
    cam = cv2.VideoCapture(0)
    if (cam.isOpened() == False): 
      print("Error reading video")
    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))
    size = (frame_width, frame_height)
    input("Press enter to take background video")
    result = cv2.VideoWriter('./background_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    for i in range(60):
      ret, frame = cam.read()
      if ret == True: 
          result.write(frame)
          cv2.imshow('Frame', frame)
          if cv2.waitKey(1) & 0xFF == ord('s'):
              break
      else:
          break
    # for background video, create the background image
    capture = cv2.VideoCapture('./background_video.avi')
    frameIds = capture.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
    frames = []
    for frame_id in frameIds:
      capture.set(cv2.CAP_PROP_FRAME_COUNT, frame_id)
      ret, frame = capture.read()
      frames.append(frame)
    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    cv2.imwrite('./background_frame.jpg', median_frame)
    cam.release()
    result.release()
    cv2.destroyAllWindows()
	
  def do_frame(self, arg):
    """
    Based on background, take short video with subject and run subtraction
    """
    cam = cv2.VideoCapture(0)
    if (cam.isOpened() == False): 
      print("Error reading video")
    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))
    size = (frame_width, frame_height)
    input("Press enter to take video with you in it")
    result = cv2.VideoWriter('./foreground_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
    for i in range(60):
      ret, frame = cam.read()
      if ret == True: 
          result.write(frame)
          cv2.imshow('Frame', frame)
          if cv2.waitKey(1) & 0xFF == ord('s'):
              break
      else:
          break
    cam.release()
    result.release()
    cv2.destroyAllWindows()


  def do_xy(self, arg):
    """
    For each frame in background and foreground, subtract, Canny edge detection, find contours, get x-y
    """
    back = cv2.VideoCapture('./background_video.avi')
    fore = cv2.VideoCapture('./foreground_video.avi')
    back_frames = []
    fore_frames = []
    diff_frames = []
    for i in range(60):
      _, back_frame = back.read()
      _, fore_frame = fore.read()
      back_gray = cv2.cvtColor(back_frame, cv2.COLOR_BGR2GRAY)
      fore_gray = cv2.cvtColor(fore_frame, cv2.COLOR_BGR2GRAY)
      back_frames.append(back_gray)
      fore_frames.append(fore_gray)
    threshold = 20
    for i in range(0, 5):
      print("frame: " + str(i))
      f_frame = fore_frames[i]
      b_frame = back_frames[i]
      diff_frame = np.zeros(shape=(720, 1280))
      for x in range(0, 720):
        for y in range(0, 1280):
          pixel = int(f_frame[x, y]) - int(b_frame[x,y])
          if abs(pixel) > threshold:
            diff_frame[x, y] = 1
      diff_frames.append(diff_frame)
    avg_img = np.mean(diff_frames, axis=0)
    labeled_image, nb_labels = ndimage.label(diff_frames[-1], structure=np.ones((3,3)))
    sizes = ndimage.sum(diff_frames[-1], labeled_image, range(nb_labels + 1))
    sizes = list(sizes)
    main_label = max(sizes)
    res_list = [i for i, value in enumerate(sizes) if value == main_label]
    print(res_list)
    main_label = res_list[0]
    for x in range(0, 720):
      for y in range(0, 1280):
        if labeled_image[x, y] != main_label:
          labeled_image[x, y] = 0
        else:
          labeled_image[x, y] = 255
    cv2.imwrite('./labels.jpg', labeled_image)
    print(labeled_image)
    image_bw = ndimage.binary_fill_holes(labeled_image).astype(int)
    cv2.imwrite('./label_bw_filled_in.jpg', image_bw)

    while True:
      cv2.imshow('Gray image', diff_frames[-1])
      k = cv2.waitKey(33)
      if k==27:    # Esc key to stop
          break
    cv2.destroyAllWindows()


  def do_drivexy(self, arg):
    '''
    Drives the digital signal to send the laser to
    a specific point. Can be called <=800 times a second.
    Uses a list of x and y points.
    '''
    
    pass

  def do_circle(self, arg):
    """
    Drives motor to make a circle
    """
    

  def do_colorpick(self, arg):
    """
    Drives the digital signal to achieve the correct
    RGB color on the white laser
    """
    pass

  def do_threshold(self, arg):
    """
    Test thresholding to determine outline points
    """
    
    pass

  def do_bye(self, arg):
    """
    Stop command line interface
    """
    print('thanks for using scalp')
    self.close()
    return True

  def close(self):
    if self.file:
        self.file.close()
        self.file = None


if __name__ == '__main__':
        c = ScALP()
        sys.exit(c.cmdloop())
