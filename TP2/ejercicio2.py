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
titlesO = []
for i in range(1,13):
    titlesO.append("Auto " + str(i))

# Autos
# imshow(imgs[n])
subplot12(imgs,titlesO)


# 2 # Pasamos a escala de grises
grays = []
for img in imgs:
    grays.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

# 2.1 # Visualizamos el resultado
titlesG = []
for i in range(1,13):
    titlesG.append("Escala de grises Auto " + str(i))

# Autos
# imshow(grays[n])
subplot12(grays,titlesG)

# 3 # Calculamos manualmente ROIs para visualizar mejor la zona de interes (patentes)
coords = [(130, 130 + 50,325,325 + 100), #(x1,x2,y1,y2)
                  (190, 190 + 50,305,305 + 100),
                  (170, 170 + 50,225,225 + 100),
                  (170, 170 + 50,305,305 + 100),
                  (105, 105 + 50,240,240 + 100),
                  (170, 170 + 50,285,285 + 100),
                  (150, 150 + 50,235,235 + 100),
                  (195, 195 + 50,175,175 + 100),
                  (220, 220 + 50,355,355 + 100),
                  (240, 240 + 50,280,280 + 100),
                  (190, 190 + 50,240,240 + 100), 
                  (175, 175 + 50,305,305 + 100)] 

# 3.1 # Definimos funcion para facilitar los recortes
def recortes(imgs, coords = None):
    """Recibe una lista de imagenes y las coordenadas de los ROIs de las imagenes
    y devuelve una lista con las imagenes recortadas en las ROIs correspondientes"""
    if coords is None:
        coords = [(130, 130 + 50,325,325 + 100), #(x1,x2,y1,y2)
                  (190, 190 + 50,305,305 + 100),
                  (170, 170 + 50,225,225 + 100),
                  (170, 170 + 50,305,305 + 100),
                  (105, 105 + 50,240,240 + 100),
                  (170, 170 + 50,285,285 + 100),
                  (150, 150 + 50,235,235 + 100),
                  (195, 195 + 50,175,175 + 100),
                  (220, 220 + 50,355,355 + 100),
                  (240, 240 + 50,280,280 + 100),
                  (190, 190 + 50,240,240 + 100), 
                  (175, 175 + 50,305,305 + 100)] 
    recortes = []
    for i,c in enumerate(coords):
        x1, x2, y1, y2 = c[0], c[1], c[2], c[3]
        img = imgs[i]
        recortes.append(img[x1:x2, y1:y2])
    return recortes

# 3.2 # Visualizamos el resultado
titlesPG = []
for i in range(1,13):
    titlesPG.append("Recorte Patente" + str(i))

# Patentes
# imshow(patentes[n])
patentes = recortes(grays)
subplot12(patentes,titlesPG)

# 4 #Aplicamos filtro high boost para resaltar caracteres de las patentes

grays_hb = []
#Seteamos los parametros del filtro
A = 2 
w = -np.ones((5,5))/(5*5)
w[2,2] = (25*A-1)/25
for gray in grays:
    grays_hb.append(cv2.filter2D(gray,-1,w))

# 4.2 # Visualizamos el resultado
titlesHB = []
for i in range(1,13):
    titlesHB.append("High boost Auto " + str(i))

titlesPHB = []
for i in range(1,13):
    titlesPHB.append("High boost Patente " + str(i))

# Autos
# imshow(grays_hb[n])
subplot12(grays,titlesG) 
subplot12(grays_hb,titlesHB) #Generar los 2 subplot 12, maximizar las pestañas, intercambiar y ver el resultado

# Patentes
patentes1 = recortes(grays)
patentes2 = recortes(grays_hb)
# imshow(patentes2[n])
subplot12(patentes1,titlesPG) 
subplot12(patentes2,titlesPHB) #Generar los 2 subplot 12, maximizar las pestañas, intercambiar y ver el resultado


# 5 # Umbralizamos la imagen con el threshold adecuado

TH1 = 145 # Justo para detectar todos los caracteres sin que se rompan
imgs_th = []

for gray in grays_hb:
    _, img_bin = cv2.threshold(gray, TH1, 1, cv2.THRESH_BINARY)
    imgs_th.append(img_bin)

# 5.1 # Visualizamos el resultado

titlesTH = []
for i in range(1,13):
    titlesTH.append("Auto " + str(i) + " - Threshold > " + str(TH1))

titlesPTH = []
for i in range(1,13):
    titlesPTH.append("Patente " + str(i) + " - Threshold > " + str(TH1))

# Autos
# imshow(imgs_th[n])
subplot12(grays_hb,titlesHB)
subplot12(imgs_th,titlesTH) #Generar los 2 subplot 12, maximizar las pestañas, intercambiar y ver el resultado

# Patentes
patentes1 = recortes(grays_hb)
patentes2 = recortes(imgs_th)

subplot12(patentes1,titlesPHB)
subplot12(patentes2,titlesPTH) #Generar los 2 subplot 12, maximizar las pestañas, intercambiar y ver el resultado


# 6 # Filtramos por area las componentes conectadas

imgs_th_fA_fRA = [img.copy() for img in imgs_th]
for img in imgs_th_fA_fRA:
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)  
    TH1 = 18 #Justo para detectar bien todos los caracteres (con 1 mas pierdo caracteres)
    TH2 = 90 #Justo para detectar bien todos los caracteres (con 1 menos pierdo caracteres)
    #Relacion de aspecto de los caracteres # VER SI HACER POR SEPARADO; EN 2 PASOS (mas eficiente es asi como esta)
    RA1 = 1.3
    RA2 = 2.4
    for i in range(num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        ancho = stats[i, cv2.CC_STAT_WIDTH]
        alto = stats[i, cv2.CC_STAT_HEIGHT]
        relAsp = alto / ancho
        if (area < TH1 or area > TH2) or (relAsp < RA1 or relAsp > RA2):
            img[labels==i] = 0


# 6.1 # Visualizamos el resultado

titles_fA_fRA = []
for i in range(1,13):
    titles_fA_fRA.append(str(TH1) + " > Area > " + str(TH2) + " - " + str(RA1) + " > Rel. Asp > " + str(RA2))

# Autos
subplot12(imgs_th,titlesTH)
subplot12(imgs_th_fA_fRA,titles_fA_fRA) #Generar los 2 subplot 12, maximizar las pestañas, intercambiar y ver el resultado

# Patentes
patentes1 = patentes2
patentes2 = recortes(imgs_th_fA_fRA)

subplot12(patentes1,titlesPTH)
subplot12(patentes2,titles_fA_fRA) #Generar los 2 subplot 12, maximizar las pestañas, intercambiar y ver el resultado

# 7 # Calculamos nuevamentes las componentes conectadas y filtramos por grupos
#  de 3 componentes cercanas horizontalmente y verticalmente
imgs_th_fA_fRA_fG = [img.copy() for img in imgs_th_fA_fRA]
for img in imgs_th_fA_fRA_fG:
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    #Ordeno stats segun la coordenada izquierda de menor a mayor
    sorted_indices = np.argsort(stats[:, cv2.CC_STAT_LEFT])  # Índices que ordenan por la coordenada izquierda
    stats = stats[sorted_indices]  # Ordeno stats según la coordenada izquierda
    new_labels = np.zeros_like(labels)  # Creamos una nueva matriz de etiquetas del mismo tamaño
    for i, idx in enumerate(sorted_indices):
        new_labels[labels == idx] = i
    # Defino umbral de distancia
    dist_max_h = 13 #13 va bien, faltan 2 patentes que quedaron incompletas, INVESTIGAR POR SEPARADO EN DETALLE
    dist_max_v = 5
    # Lista para guardar los grupos de 3 componentes cercanos
    indices = []
    intentos = 3
    while len(indices) < 6 and intentos >= 0: #Si no obtengo 6 caracteres bajo el umbral de distancia horizontal durante 5 intentos
        print(dist_max_h)
        for i in range(num_labels - 2):  # Verificamos grupos de 3
            i2 = i + 1
            i3 = i + 2
            x1 = stats[i, cv2.CC_STAT_LEFT]
            x2 = stats[i2, cv2.CC_STAT_LEFT]
            x3 = stats[i3, cv2.CC_STAT_LEFT]
            y1 = stats[i, cv2.CC_STAT_TOP]
            y2 = stats[i2, cv2.CC_STAT_TOP]
            y3 = stats[i3, cv2.CC_STAT_TOP]
            distH_1_2 = x2 - x1 <= dist_max_h
            distH_2_3 = x3 - x2 <= dist_max_h
            distV_1_2 = abs(y2 - y1) <= dist_max_v  
            distV_2_3 = abs(y3 - y2) <= dist_max_v
            # Verificar si las distancias entre componentes consecutivos son menores al umbral
            if distH_1_2 and distH_2_3 and distV_1_2 and distV_2_3 and i not in indices:
                indices += [i, i2, i3]
            if len(indices) == 6: #Si obtengo 2 grupos de 3 componentes(caracteres) termino
                break
        dist_max_h -= 1
        intentos -= 1
    # Elimino las componentes que no correspondan a los indices de los grupos de 3
    for i in range(num_labels):
        # print(i)
        if i not in indices:
            print(i)
            img[new_labels==i] = 0

# 7.1 # Visualizamos el resultado
titles_fA_fRA_fG = []
for i in range(1,13):
    titles_fA_fRA_fG.append("Auto " + str(i) + "- Filtro grupos 3 cercanos")

titlesP_fA_fRA_fG = []
for i in range(1,13):
    titlesP_fA_fRA_fG.append("Patente " + str(i) + "- Filtro grupos 3 cercanos")

# Autos
subplot12(imgs_th_fA_fRA,titles_fA_fRA)
subplot12(imgs_th_fA_fRA_fG,titles_fA_fRA_fG)

# Patentes
patentes1 = recortes(imgs_th_fA_fRA)
patentes2 = recortes(imgs_th_fA_fRA_fG)
subplot12(patentes1,titles_fA_fRA)
subplot12(patentes2,titlesP_fA_fRA_fG)

# 8 # Calculamos nuevamente componentes conectadas, guardamos coordenadas y 
# creamos bounding box en imagen original
coords2 = []
k = 3
h = 1
r = 10

# Identificacion de patentes en imagen original
identificacion = [img.copy() for img in imgs]

# Identifiacion de caracteres de patentes binarios
caracteres = [img.copy() for img in imgs_th_fA_fRA_fG]
for i, img in enumerate(caracteres):
    caracteres[i] = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB)


for i,img in enumerate(imgs_th_fA_fRA_fG):
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    sorted_indices = np.argsort(stats[:, cv2.CC_STAT_LEFT])  # Índices que ordenan por la coordenada izquierda
    stats = stats[sorted_indices]
    # Guardamos la coordenada del punto medio derecho de la primer letra 
    x1 = stats[1, cv2.CC_STAT_LEFT] - k
    y1 = stats[-1, cv2.CC_STAT_TOP] - k
    argmax = np.argmax(stats[:, cv2.CC_STAT_LEFT])
    x2 = stats[argmax, cv2.CC_STAT_LEFT] + stats[argmax, cv2.CC_STAT_WIDTH] + k
    argmax = np.argmax(stats[:, cv2.CC_STAT_TOP]) 
    y2 = stats[argmax, cv2.CC_STAT_TOP] + stats[argmax, cv2.CC_STAT_HEIGHT] + k
    # Resaltamos los caracteres con el high boost que hicimos anteriormente
    recorte_gris = grays_hb[i][y1:y2,x1:x2]
    recorte_rgb = cv2.cvtColor(recorte_gris, cv2.COLOR_GRAY2RGB)
    # Pegar el recorte en la imagen RGB en la posición deseada
    identificacion[i][y1:y2, x1:x2] = recorte_rgb
    cv2.rectangle(identificacion[i], (x1, y1), (x2, y2), color=(0,255,0), thickness=2)
    cv2.rectangle(caracteres[i], (x1, y1), (x2, y2), color=(255,255,255), thickness=1)
    # Calculo las componentes conectadas para obtener caracteres y hacer bounding box
    i_char = 1
    for st in stats[1:]:
        yc1, xc1, yc2, xc2 = st[1] - h, st[0] - h, st[1]+st[3] + h, st[0]+st[2] + h
        cv2.rectangle(identificacion[i], (xc1, yc1), (xc2, yc2), color=(255,0,0), thickness=1)
        pos = (xc1 + (xc2-xc1)//2 - 3, yc1-2)
        text = str(i_char)
        i_char += 1
    caracteres[i] = caracteres[i][y1-r:y2+r,x1-r:x2+r]
    coord = (y1-r,y2+r,x1-r,x2+r) #Agrando para obtener un recorte mas prolijo y automatico sin usar coords manual
    coords2.append(coord) # Invertido para usar con funcion recortes

    
subplot12(caracteres, titlesO)
patentes = recortes(identificacion, coords2) #Estas coordenadas son automaticas, no manuales como coord
subplot12(patentes, titlesO)


for i, img in enumerate(imgs):
    plt.figure()
    plt.subplot(221); imshow(img, new_fig=False, title="Auto " + str(i))
    x1, x2, y1, y2 = coords2[i]
    plt.subplot(222); imshow(identificacion[i], new_fig=False, title="Deteccion patente " + str(i))
    plt.subplot(223); imshow(patentes[i], new_fig=False, title="Patente " + str(i))
    plt.subplot(224); imshow(caracteres[i], new_fig=False, title="Caracteres " + str(i))
    plt.show(block=False)


for img in grays_hb:
    cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)