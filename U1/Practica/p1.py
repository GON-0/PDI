import cv2
import numpy as np
import matplotlib.pyplot as plt

#1)--------------------------------------------------------
#Cargo imagen
img = cv2.imread('img_calculadora.tif', cv2.IMREAD_GRAYSCALE)

#2)--------------------------------------------------------
#Info
img.dtype
h,w = img.shape
h,w

#3)--------------------------------------------------------
#Stats
img.min()
img.max()

#4)--------------------------------------------------------
np.unique(img)
len(np.unique(img))

#Visualizacion
plt.imshow(img,cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title('Calculadora')
plt.show(block=False)

#5)--------------------------------------------------------
unique, counts = np.unique(img, return_counts=True)
unique
counts
max = np.max(counts)
min = np.min(counts)
argsmin = np.argwhere(counts == min)
argsmax = np.argwhere(counts == max)

#Nivel de gris con menor frecuencia
unique[argsmin]
#Nivel de gris con mayor frecuencia
unique[argsmax]

#6)--------------------------------------------------------

#Coordenadas eje y
ymin_sin = ymin_cos = ymin_tan = 322
ymax_sin = ymax_cos = ymax_tan = 443 

#Coordenadas eje x
xmin_sin = 725
xmax_sin = 885

xmin_cos = 950
xmax_cos = 1110

xmin_tan = 1175
xmax_tan = 1335

img_crop_sin = img[ymin_sin:ymax_sin,xmin_sin:xmax_sin]
img_crop_cos = img[ymin_cos:ymax_cos,xmin_cos:xmax_cos]
img_crop_tan = img[ymin_tan:ymax_tan,xmin_tan:xmax_tan]

#Visualizacion:

plt.subplot(1,3,1)
plt.imshow(img_crop_sin,cmap='gray', vmin=0, vmax=255)
plt.title('Seno')
plt.show(block=False)

plt.subplot(1,3,2)
plt.imshow(img_crop_cos,cmap='gray', vmin=0, vmax=255)
plt.title('Coseno')
plt.show(block=False)

plt.subplot(1,3,3)
plt.imshow(img_crop_tan,cmap='gray', vmin=0, vmax=255)
plt.title('Tangente')
plt.show(block=False)

#7)--------------------------------------------------------

img_copy = img.copy()

img_copy[ymin_sin:ymax_sin,xmin_sin:xmax_sin] = img_crop_tan
img_copy[ymin_tan:ymax_tan,xmin_tan:xmax_tan] = img_crop_sin

plt.subplot(1,2,1)
plt.imshow(img,cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.show(block=False)

plt.subplot(1,2,2)
plt.imshow(img_copy,cmap='gray', vmin=0, vmax=255)
plt.title('Modificada')
plt.show(block=False)

#8)--------------------------------------------------------

#1 - Recorto enter

enter = img[540:665,50:435].copy()

plt.imshow(enter,cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title('Enter')
plt.show(block=False)

#2 - Recorto bordes laterales
xmin_borde_izq = 0
xmax_borde_izq = 20

xmin_borde_der = 365
xmax_borde_der = 385

ymin = 0
ymax = 125

borde_izq = enter[ymin:ymax, xmin_borde_izq:xmax_borde_izq]
borde_der = enter[ymin:ymax, xmin_borde_der:xmax_borde_der]

plt.subplot(1,2,1)
plt.imshow(borde_izq,cmap='gray', vmin=0, vmax=255)
plt.title('Borde enter izquierdo')
plt.show(block=False)

plt.subplot(1,2,2)
plt.imshow(borde_der,cmap='gray', vmin=0, vmax=255)
plt.title('Borde enter derecho')
plt.show(block=False)


#3 - Pego los bordes mas cerca de la palabra "ENTER"
xmin_reubicado_izq = 55
xmax_reubicado_izq = 75

xmin_reubicado_der = 306
xmax_reubicado_der = 326


enter[ymin:ymax,xmin_reubicado_izq:xmax_reubicado_izq] = borde_izq
enter[ymin:ymax,xmin_reubicado_der:xmax_reubicado_der] = borde_der

enter = enter[ymin:ymax,55:326]

plt.imshow(enter,cmap='gray', vmin=0, vmax=255)
plt.title('ENTER más pequeño')
plt.show(block=False)


#4 - Mejoro tamaño del enter con resize
enter.shape
enter_resize = cv2.resize(enter,(170,125))

#5 - Calculo las coordenadas y pego encima del sin, centrado
xmin_enter_re = 715
xmax_enter_re = 885

ymin_enter_re = 320
ymax_enter_re = 445

img2 = img.copy()

img2[ymin_enter_re:ymax_enter_re,xmin_enter_re:xmax_enter_re] = enter_resize

plt.imshow(img2,cmap='gray', vmin=0, vmax=255)
plt.title('Calculadora modificada')
plt.show(block=False)


#8)--------------------------------------------------------

#1 - Calculo coordenadas

#Creo una copia para no modificar la original
img2 = img.copy()

xmin_456 = xmin_789 = 300
xmax_456 = xmax_789 = 1060

ymin_456 = 1010
ymax_456 = 1110

ymin_789 = 785
ymax_789 = 885

n456 = img2[ymin_456:ymax_456,xmin_456:xmax_456]
n789 = img2[ymin_789:ymax_789,xmin_789:xmax_789]


# 2 - Pinto las etiquetas numericas con nivel de gris 170
n456[n456 > 200] = 170                               
n789[n789 > 200] = 170                               

plt.subplot(1,2,1)
plt.imshow(n456, cmap='gray', vmin=0, vmax=255)
plt.title('Numeros 4, 5 y 6')

plt.subplot(1,2,2)
plt.imshow(n789, cmap='gray', vmin=0, vmax=255)
plt.title('Numeros 7, 8 y 9')
plt.show(block=False)

# 3 - No hace falta pegarlos nuevamente porque estan referenciados
#La transformacion se ve reflejada en img2

#Visualizo el resultado
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen original')

plt.subplot(1,2,2)
plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen modificada')
plt.show(block=False)
