import os
import cv2

try: 
      
    # creating a folder named data 
    if not os.path.exists('Frame-data'): 
        os.makedirs('Frame-data') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of Frame-data') 
  

data = []
labels = []

for label in os.listdir('.\\Training Data\\'):
  folderPath = os.path.join('.\\Training Data\\', label) # Getting each folder path by looping through the sub-directories
  if not os.path.isdir(folderPath): # because this file is included in that directory
    continue

  for imagePath in os.listdir(folderPath):
    filePath = os.path.join(folderPath,imagePath) # Then getting the file path for cv2
    cam = cv2.VideoCapture(filePath)
    data.append(cam)
    labels.append(label)
    

print(data)

currentframe = 0

for i,x in enumerate(data):
    # frame 
    while(True): 
        # reading from frame 
        ret,frame = x.read() 
        if not x.isOpened():
            print(f"Failed to open video: {filePath}")  # Debugging output
            break  # Skip this file if it can't be opened

        if ret: 
            # if video is still left continue creating images 

            if frame is None:
               print("Frame does not exist!")
               break
            else:
                name = f'.\\Frame-data\\{labels[i]}\\frame' + str(currentframe) + '.jpg'
                print ('Creating...' + name) 

                # writing the extracted images 
                success = cv2.imwrite(name, frame) 
                if not success:
                    print(f"Problem writing {name}")
                # increasing counter so that it will 
                # show how many frames are created 
                currentframe += 1
        else: 
            print(f"Finished Writing Frames from {labels[i]}_{i}")
            break

    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 
