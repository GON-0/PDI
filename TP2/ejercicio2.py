# Importamos librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import cv2


"""
--------------------------------------------------------------------------------
                    IDENTIFICACIÓN DE PATENTES
--------------------------------------------------------------------------------

El presente trabajo se basa en el procesamiento de 12 imagenes que contienen un auto
para detectar y segmentar automaticamente la patente del auto en cuestion

ADECUAR LO SIGUIENTE
El codigo se divide en dos partes correspondientes a las siguientes funciones:

    # monedas : Encargada de clasificar las monedas, identificarlas e informar el resultado
    # dados : Encargada de identificar los dados, su valor e informar el resultado

Por último, se obtiene un resultado final con la combinacion de la salida de 
ambas funciones, identificando monedas y dados en una imagen final
"""

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    """Definimos funcion para simplificar codigo de visualizaciones"""
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

def subplot12(imgs, titles, share = False):
    """Recibe 12 imagenes, 12 titulos y las muestra en un subplot identificando
       cada uno de los autos, del 1 al 12"""
    plt.figure()
    if share:
        ax1 = plt.subplot(3,4,1); imshow(imgs[0], new_fig=False, title=titles[0])
    else:
        plt.subplot(3,4,1); imshow(imgs[0], new_fig=False, title=titles[0])
    for i in range(1,12):
        if share:
            plt.subplot(3,4,i+1, sharex=ax1, sharey=ax1); imshow(imgs[i], new_fig=False, title=titles[i])
        else:
            plt.subplot(3,4,i+1); imshow(imgs[i], new_fig=False, title=titles[i])
    plt.show(block=False)

n = 0 # Numero de auto para ir visualizando

# 1 # Cargamos las imagenes y las almacenamos en una lista
imgs = []
for i in range(1,13):
    img = cv2.imread("img" + str(i).zfill(2) + ".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)

# 1.1 # Visualizamos el resultado
titles = []
for i in range(1,13):
    titles.append("Auto " + str(i))
# imshow(imgs[n])
subplot12(imgs,titles)


# 2 # Pasamos a escala de grises
grays = []
for img in imgs:
    grays.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

# 2.1 # Visualizamos el resultado
titles = []
for i in range(1,13):
    titles.append("Escala de grises Auto " + str(i))
# imshow(grays[n])
subplot12(grays,titles)

# 3 # Calculamos manualmente ROIs para visualizar mejor la zona de interes (patentes)

# 3.1 # Definimos funcion para facilitar los recortes
def recortes(imgs, coords = None):
    """Recibe una lista de imagenes y las coordenadas de los ROIs de las imagenes
    y devuelve una lista con las imagenes recortadas en las ROIs correspondientes"""
    if coords is None:
        coords = [(130,325),
                  (190,305),
                  (170,225),
                  (170,305),
                  (105,240),
                  (170,285),
                  (150,235),
                  (195,175),
                  (220,355),
                  (240,280),
                  (190,240), 
                  (175,305)] 
    recortes = []
    alto = 50
    ancho = 100
    for i,c in enumerate(coords):
        x1, y1 = c[0], c[1]
        x2, y2 = x1 + alto, y1 + ancho
        img = imgs[i]
        recortes.append(img[x1:x2, y1:y2])
    return recortes

# 3.2 # Visualizamos el resultado
titles = []
for i in range(1,13):
    titles.append("Escala de grises Patente" + str(i))
patentes = recortes(grays)
# imshow(patentes[n])
subplot12(patentes,titles)

# 4 #Aplicamos filtro high boost para resaltar caracteres de las patentes

grays_hb = []
#Seteamos los parametros del filtro
A = 2 
w = -np.ones((5,5))/(5*5)
w[2,2] = (25*A-1)/25
for gray in grays:
    grays_hb.append(cv2.filter2D(gray,-1,w))

# 4.2 # Visualizamos el resultado
titles = []
for i in range(1,13):
    titles.append("High boost Patente " + str(i))
patentes1 = recortes(grays)
patentes2 = recortes(grays_hb)
# imshow(patentes[n])
subplot12(patentes1,titles)
subplot12(patentes2,titles)


# 5 # Umbralizamos la imagen con el threshold adecuado

TH1 = 145
imgs_th = []

for gray in grays_hb:
    _, img_bin = cv2.threshold(gray, TH1, 1, cv2.THRESH_BINARY)
    imgs_th.append(img_bin)

titles = []
for i in range(1,13):
    titles.append("Auto " + str(i) + " - Threshold > " + str(TH1))

# imshow(imgs_th[n])
subplot12(imgs_th,titles)
patentes = recortes(imgs_th)
subplot12(patentes,titles)



imgs_th_fA = [img.copy() for img in imgs_th]
for img in imgs_th_fA:
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)  
    TH1 = 18 #Justo para detectar bien todos los caracteres (con 1 mas pierdo caracteres)
    TH2 = 90 #Justo para detectar bien todos los caracteres (con 1 menos pierdo caracteres)
    for i in range(num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < TH1 or area > TH2:
            img[labels==i] = 0

titles = []
for i in range(1,13):
    titles.append("Auto " + str(i) + " Filrado por area entre [" + str(TH1) + ", " + str(TH2) + "]" )

subplot12(imgs_th_fA,titles)
patentes = recortes(imgs_th_fA)
subplot12(patentes,titles)



ax1 = plt.subplot(121); imshow(imgs_th1[n], new_fig=False, title="")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(imgs_th1_filtA[n], new_fig=False, title="")
plt.show(block=False)

