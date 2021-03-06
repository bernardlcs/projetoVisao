import numpy as np
import scipy

import cv2
import os
import csv
import time

path = 'C:/projetos/visao/face'
os.chdir(path)

#Salva o arquivo com titulo atualizado por data e hora
titulo = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")
saida = open('face_recon_'+titulo+'.csv', 'w')
export = csv.writer(saida, quoting=csv.QUOTE_NONNUMERIC)


file_list = []

for file in os.listdir(path):
    if file.endswith(".jpg"):
        file_list.append(file)


for file in file_list:
    #Para cada arquivo na lista
    #Estabelece os classificadores de face
    face_cascade = cv2.CascadeClassifier('C:/opne3/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    face_alt_cascade = cv2.CascadeClassifier('C:/opne3/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml')
    face_alt2_cascade = cv2.CascadeClassifier('C:/opne3/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face_alt_tree_cascade = cv2.CascadeClassifier('C:/opne3/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml')
    
    #ler a imagem e converte para escala de cinza
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Faz as classificacoes
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    faces2 = face_alt_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    faces3 = face_alt2_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    faces4 = face_alt_tree_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    
    #Organiza numa as classificacoes numa lista para loop
    classifiers = [faces, faces2, faces3, faces4]
    
    # Coloca os quadrados nas faces
    for classifier in classifiers:
        for (x,y,w,h) in classifier:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        
        print("Para a imagem "+file+", foram encontradas {0} faces!".format(len(classifier)))
        
        #Exibe as imagens com retangulos
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Exibe a media, variancia e moda de cada classificador
    encontrados = []
    
    for (classifier) in classifiers:
        x = format(len(classifier))
        encontrados.append(x)
    
    encontrados = np.asarray(encontrados, dtype = np.float16)
    media = np.mean(encontrados)
    variancia = np.var(encontrados)
    
    
    if file == file_list[0]:
        export.writerow(["imagem","media","variancia"])
        export.writerow([file, media, variancia])
    else:
        export.writerow([file, media, variancia])

saida.close()

print ("fim")
