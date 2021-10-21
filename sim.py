from qibullet import SimulationManager
import pybullet as p
from qibullet import PepperVirtual
import cv2
from tensorflow.keras import models
import numpy
from PIL import Image
import os,glob
import time
import random

def model_prediction(img,x, y, w, h) :
	cropedImg = img[y+2:y+h-2, x+2:x+w-2]
	cropedImg = cv2.resize(cropedImg, (300, 300), interpolation = cv2.INTER_AREA)
	cropedImg = numpy.asarray(cropedImg)
	cropedImg = numpy.expand_dims(cropedImg, 0)

	cropedImg = cropedImg/ 255.0

	y_pred=model.predict(cropedImg)

	prediction_string = ""

	print(y_pred)

	if(y_pred[0][0] >= y_pred[0][1]):
		prediction_string = "zemmour p=" + str(y_pred[0][0])
	else:
		prediction_string = "chalamet p=" + str(y_pred[0][1])

	return prediction_string


if __name__ == "__main__":
	simulation_manager = SimulationManager()
	client_id = simulation_manager.launchSimulation(gui=True)


#######################################################
# create texture
#######################################################

# Définir les chemins vers les images à prédire,
images_path = glob.glob('chalamet_zemmour_new/chalamet_zemmour_new_300x300/*.jpg')
template_path = 'texture_template.jpg'

i = 0
# Chargement des images à prédire
for images in images_path :
   img = Image.open(images)
   template = Image.open(template_path)

   template.paste(img,(220,607))
   template.save("pictures_sim/texture_"+str(i)+".jpg")
   i = i + 1



#######################################################
# create pictures
#######################################################

random_id1 = random.randint(0, 2)
random_id2 = random.randint(3, 5)

texture_1 = p.loadTexture("./pictures_sim/texture_"+str(random_id1)+".jpg")
texture_2 = p.loadTexture("./pictures_sim/texture_"+str(random_id2)+".jpg")

p.connect(p.DIRECT)
picture_visual1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[1,0.01,1])
picture_visual2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[1,0.01,1])

picture1 = p.createMultiBody(baseVisualShapeIndex=picture_visual1, basePosition = [5, 1.5, 1], baseOrientation = [1,1,0,0])
picture2 = p.createMultiBody(baseVisualShapeIndex=picture_visual2, basePosition = [5, -1.5, 1], baseOrientation = [1,1,0,0])

p.changeVisualShape(picture_visual1, -1, textureUniqueId=texture_1)
p.changeVisualShape(picture_visual2, -1, textureUniqueId=texture_2)


#######################################################
# create robot
#######################################################

pepper = simulation_manager.spawnPepper(
client_id,
spawn_ground_plane=True)

handle = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_TOP)




#######################################################
# face detection & model prediction setup
#######################################################

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = models.load_model('model_keras')

#######################################################
# thread
#######################################################

prediction_string = ""

try:
	while True:

		img = pepper.getCameraFrame(handle)
		

		detected_faces = face_cascade.detectMultiScale(image=img, scaleFactor=1.3, minNeighbors=6)  #paramètres à modifier

		if(len(detected_faces) > 0) :

			for face in detected_faces :

				(x, y, w, h) = face
				cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)

				prediction_string = model_prediction(img, x, y, w, h)
				time.sleep(1)
				cv2.putText(img, prediction_string, (x, y-12),  cv2.FONT_HERSHEY_SIMPLEX,  0.4,(0, 255, 0),1)

		cv2.imshow("top camera", img)
		cv2.waitKey(1)


except KeyboardInterrupt:
	simulation_manager.stopSimulation(client)