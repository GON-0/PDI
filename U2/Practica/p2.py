import cv2
import numpy as np
import matplotlib.pyplot as plt

"""---------------------EJERCICIO 1---------------------
A partir de la imagen "faces.jpg" aplicar un efecto de borrosidad en las caras"
"""

"""1.1 Carga de la imagen de entrada"""

"""a)---"""
img = cv2.imread('faces.jpg')
plt.imshow(img), plt.show(block=False)

#La imagen esta en BGR, paso a RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img), plt.show(block=False)

"""b)---"""
img.dtype
img.shape
#Maximo para cada canal (C1,C2,C3 - RGB respec)
img[:,:,0].max(), img[:,:,1].max(), img[:,:,2].max()
#Minimo para cada canal (C1,C2,C3 - RGB respec)
img[:,:,0].min(), img[:,:,1].min(), img[:,:,2].min()

"""1.2 Filtrado manual"""

"""c)---"""

#Coordenadas de las caras (manual)
#(ysi,xsi,yid,xid)
cara1 = (120,54,165,122)
cara2 = (250,17,295,85)
cara3 = (381,45,424,95)
cara4 = (473,73,516,135)

#Para ir graficando las caras y ver si estan bien las medidas
cara = cara1
subface = img[cara[1]:cara[3],cara[0]:cara[2]] #rangox : rangoy
plt.imshow(subface), plt.show(block=False)

img2 = img.copy()
#Podria hacer lo siguiente con un for pero quedo asi
cv2.rectangle(img2, (cara1[0],cara1[1]), (cara1[2],cara1[3]), (255,0,0))
cv2.rectangle(img2, (cara2[0],cara2[1]), (cara2[2],cara2[3]), (255,0,0))
cv2.rectangle(img2, (cara3[0],cara3[1]), (cara3[2],cara3[3]), (255,0,0))
cv2.rectangle(img2, (cara4[0],cara4[1]), (cara4[2],cara4[3]), (255,0,0))
plt.imshow(img2), plt.show(block=False)

#Notar que a diferencia de la indexacion de matriz,
#rectangle pide puntos de la forma (y,x) (con y eje horizontal e x vertical)
#y al indexar se hace de la siguiente forma img[rangox,rangoy]
"""d)---"""

#Recortamos cada cara
img3 = img.copy()
subface1 = img3[cara1[1]:cara1[3],cara1[0]:cara1[2]]
subface2 = img3[cara2[1]:cara2[3],cara2[0]:cara2[2]]
subface3 = img3[cara3[1]:cara3[3],cara3[0]:cara3[2]]
subface4 = img3[cara4[1]:cara4[3],cara4[0]:cara4[2]]

plt.subplot(221)
plt.imshow(subface1)
plt.subplot(222)
plt.imshow(subface2)
plt.subplot(223)
plt.imshow(subface3)
plt.subplot(224)
plt.imshow(subface4)
plt.show()

"""e)---"""
k = 15

#Para mejorar el codigo se puede hacer en un for lo siguiente


# subface1 = cv2.blur(subface1,(k,k))
# subface2 = cv2.blur(subface2,(k,k))
# subface3 = cv2.blur(subface3,(k,k))
# subface4 = cv2.blur(subface4,(k,k))

subface1 = cv2.GaussianBlur(subface1,(k,k),10)
subface2 = cv2.GaussianBlur(subface2,(k,k),10)
subface3 = cv2.GaussianBlur(subface3,(k,k),10)
subface4 = cv2.GaussianBlur(subface4,(k,k),10)



plt.subplot(221)
plt.imshow(subface1)
plt.subplot(222)
plt.imshow(subface2)
plt.subplot(223)
plt.imshow(subface3)
plt.subplot(224)
plt.imshow(subface4)
plt.show()

"""f)---"""
img3[cara1[1]:cara1[3],cara1[0]:cara1[2]] = subface1
img3[cara2[1]:cara2[3],cara2[0]:cara2[2]] = subface2
img3[cara3[1]:cara3[3],cara3[0]:cara3[2]] = subface3
img3[cara4[1]:cara4[3],cara4[0]:cara4[2]] = subface4

plt.imshow(img3), plt.show(block=False)


"""1.2 Filtrado automatico"""

"""g)---"""
grayimg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.imshow(grayimg, cmap='gray')
plt.show()


"""h)---"""
#Inicializo, cargo y ejecuto el modelo clasificador
face_cascade = cv2.CascadeClassifier()                                          
face_cascade.load(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")    
faces = face_cascade.detectMultiScale(grayimg) 

"""i)---"""
#Probamos con un ejemplo para ver como son las coordenadas y luego generalizar
xsd = faces[0][1]
xid = faces[0][1] + faces[0][3]
ysd = faces[0][0]
yid = faces[0][0] + faces[0][2]

subface = img[xsd:xid,ysd:yid]
plt.imshow(subface), plt.show(block=False)

img2 = img.copy() #La imagen en escala de grises es para el modelo detector nomas

subfaces = []
for cara in faces:
    xsd = cara[1]
    xid = cara[1] + cara[3]
    ysd = cara[0]
    yid = cara[0] + cara[2]
    subface = img[xsd:xid,ysd:yid] #Guardo las subfaces para el siguiente apartado
    subfaces.append(subface)
    #subface = grayimg[xsd:xid,ysd:yid] # para ir viendo las caras
    cv2.rectangle(img2,(ysd,xsd),(yid,xid),(255,0,0),2)

plt.imshow(img2), plt.show(block=False)

"""j)---"""

for i,cara in enumerate(subfaces):
    plt.subplot(2,2,i+1)
    plt.imshow(cara), plt.show(block=False)

plt.show()


"""k)---"""
img3 = img.copy()

subfaces = []
for i,cara in enumerate(faces):
    xsd = cara[1]
    xid = cara[1] + cara[3]
    ysd = cara[0]
    yid = cara[0] + cara[2]
    subface = img3[xsd:xid,ysd:yid]
    subface = cv2.GaussianBlur(subface,(k,k),10)
    subfaces.append(subface)
    plt.subplot(2,2,i+1)
    plt.imshow(subface), plt.show(block=False)

plt.show()

plt.imshow(subfaces[0]), plt.show()

"""l)---"""

for i,cara in enumerate(faces):
    xsd = cara[1]
    xid = cara[1] + cara[3]
    ysd = cara[0]
    yid = cara[0] + cara[2]
    img[xsd:xid,ysd:yid] = subfaces[i]
    x = 0 #Es para poner algo porque sino no se termina de ejecutar el for en consola y hay que tocar enter
    

plt.imshow(img)
plt.show(block=False)


"""---------------------EJERCICIO 1---------------------
    Quitar la mancha del documento mediante tecnicas y herramientas
    de procesamiento de imagenes
"""

img = cv2.imread("john_canny_bio.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray")
plt.show()