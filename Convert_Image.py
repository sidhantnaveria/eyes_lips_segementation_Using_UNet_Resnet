import os
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from random import randint

class Convert_Images:
    def read_images(image_path, mask_path, image_size=(256,256)):
        image_dict = {}
        mask_dict = {}
        for img in os.listdir(image_path):
            try:
              if int(img.split(".")[0])<=1999 :  
                  image_dict[img] = cv2.resize(cv2.imread(os.path.join(image_path,img)),(256,256))
            except:
              continue
            if int(img.split(".")[0])<=1999 :
                u_lip= ['{0:05}'.format(int(img.split('.',1)[0])) + '_u_lip']
                r_eye = ['{0:05}'.format(int(img.split('.',1)[0])) + '_r_eye']
                l_eye = ['{0:05}'.format(int(img.split('.',1)[0])) + '_l_eye']
                l_lip= ['{0:05}'.format(int(img.split('.',1)[0])) + '_l_lip']
                
                u_lip_image= cv2.imread(os.path.join(mask_path,u_lip[0]+'.png'))
                if not os.path.exists(os.path.join(mask_path,u_lip[0]+'.png')):
                    u_lip_image = np.zeros(image_dict[img].shape)
                r_eye_image = cv2.imread(os.path.join(mask_path,r_eye[0]+'.png'))
                if not os.path.exists(os.path.join(mask_path,r_eye[0]+'.png')):
                    r_eye_image = np.zeros(image_dict[img].shape)
                l_eye_image = cv2.imread(os.path.join(mask_path,l_eye[0]+'.png'))
                if not os.path.exists(os.path.join(mask_path,l_eye[0]+'.png')):
                    l_eye_image = np.zeros(image_dict[img].shape)
                l_lip_image = cv2.imread(os.path.join(mask_path,l_lip[0]+'.png'))
                if not os.path.exists(os.path.join(mask_path,l_lip[0]+'.png')):
                    l_lip_image = np.zeros(image_dict[img].shape)
                
                mask_dict[img] = cv2.resize(u_lip_image,(256,256))+cv2.resize(r_eye_image,(256,256))+cv2.resize(l_eye_image,(256,256))+cv2.resize(l_lip_image,(256,256))
        return (image_dict, mask_dict)
    
    def to_npy(image_dict,mask_dict):
        print("In to_npy")
        image = []
        mask = []
        for img, msk in zip(image_dict.values(), mask_dict.values()):
          image.append(img)
          mask.append(msk.astype('uint16'))
        image = np.array(image)
        mask = np.array(mask)

        mask = mask[:,:,:,1]
        mask = np.expand_dims(mask, axis=-1)
        print('Saving the images as .npy files')
        np.save('Images',image)
        np.save('Masks',mask)
        
        
images_path = os.path.join("C:/Users/sidha/Downloads/CelebAMask-HQ/CelebAMask-HQ","CelebA-HQ-img/")
mask_path = os.path.join("C:/Users/sidha/Downloads/CelebAMask-HQ/CelebAMask-HQ","CelebAMask-HQ-mask-anno/0")

# Save the images to .npy files
image_dict, mask_dict = Convert_Images.read_images(images_path, mask_path)
Convert_Images.to_npy(image_dict, mask_dict)


X = np.load(os.path.join(os.getcwd(),"Images.npy"))
Y = np.load(os.path.join(os.getcwd(),"Masks.npy"))


# print("Number of Images:", X.shape[0])
# print("Shape of Images:", X.shape[1],"x", X.shape[1])

# print("Number of Masks:", Y.shape[0])
# print("Shape of Masks:", Y.shape[1],"x", Y.shape[1])

# fig = plt.figure(figsize=(10,10))
# plt.subplots(2, 4, figsize=(10,10), sharex='row',gridspec_kw={'hspace': 0, 'wspace': 0})

# idx = randint(0, X.shape[0] - 1)

# for i in list(np.linspace(1,11,6,dtype = int)):
#   plt.subplot(2,6,i)
#   plt.imshow(X[idx + i,:,:,:])
#   plt.axis("off")
#   plt.title('Image')

#   plt.subplot(2,6,i+1)
#   plt.imshow(Y[idx + i,:,:,0])
#   plt.axis("off")
#   plt.title('Mask')

# plt.subplots_adjust(wspace=0)
# plt.tight_layout()
# plt.show()
