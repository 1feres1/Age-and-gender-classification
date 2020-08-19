import cv2
import os
import numpy as np
import argparse
from pre_trained_model import WideResNet
import matplotlib as plt

case_path = "weights/haarcascade_frontalface_alt.xml"
weight_path = "weights/weights.18-4.06.hdf5"
image_path = 'test_files/pexels-photo-415829.jpeg'
x= 'test_files/18184601-four-happy-teenage-girls-friends.jpg'


class FaceCV_image (object) :


    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV_image, cls).__new__(cls)
        return cls.instance

    def __init__(self ,depth =16 , width = 8, face_size =64):

        self.face_size = face_size
        self.model = WideResNet (face_size , depth= depth , k = width)()
        self.model.load_weights("weights/weights.18-4.06.hdf5")

    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


    def crop_face (self, image , cordonnates , margin = 40 , size = 64):

        im_hight , im_wight , _ = image.shape

        if cordonnates is None :
            cordonnates = [0,0,im_hight , im_wight]

        x,y , w, h = cordonnates

        margin = int (min (w, h ) *margin /100)

        x_a = x -margin
        y_a = y- margin
        x_b = x+ w + margin
        y_b = y+h + margin

        if x_a < 0 :
            y_b = min (x_b - x_a , im_wight -1 )
            x_a = 0

        if y_a< 0 :
            y_b = min (y_b - y_a , im_wight -1 )
            x_a = 0

        if x_b > im_wight:
            x_a = max(x_a - (x_b - im_wight), 0)
            x_b = im_wight
        if y_b > im_hight:
            y_a = max(y_a - (y_b - im_hight), 0)
            y_b = im_hight


        cropped = image[y_a : y_b , x_a : x_b]
        cropped = cv2.resize(cropped , (size,size) , interpolation= cv2.INTER_AREA)
        cropped = np.array(cropped)

        return cropped , (x_a ,y_a , x_b - x_a , y_b - y_a)


    def detect_face(self  ) :


        face_cascade = cv2.CascadeClassifier(case_path)

        image = image_path
        image = cv2.imread(image)
        image = np.array(image)
        while True :

            gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

            faces= face_cascade.detectMultiScale(gray , scaleFactor= 1.2 , minNeighbors= 10 , minSize= (self.face_size , self.face_size))

            if faces is not()  :

                face_imgs = np.empty((len(faces) ,self.face_size , self.face_size , 3))
                for i , face in enumerate(faces) :
                    face_im , cropped = self.crop_face(image , face )
                    (x,y ,w ,h)  = cropped
                    cv2.rectangle(image , (x,y ) , (x+w , y+h ) , (255,200,0) , 2)
                    face_imgs[i , :,:,:] = face_im

                if len(face_imgs) > 0:
                    results = self.model.predict(face_imgs)
                    predicted_genders = results[0]
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = results[1].dot(ages).flatten()


                for i, face in enumerate(faces):
                        label = "{}, {}".format(int(predicted_ages[i]),
                                                "F" if predicted_genders[i][0] > 0.5 else "M")
            else :
                print ('no faces')
            cv2.imshow('hhhhhh', image)
            if cv2.waitKey(5) == 27:  # ESC key press
                break


        print (1111111111111)
        cv2.waitKey(0)
        cv2.destroyAllWindows()









def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    depth = args.depth
    width = args.width

    face = FaceCV_image(depth=depth, width=width)

    face.detect_face()


if __name__ == "__main__":
    main()





        

























