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
El codigo se divide en ? partes correspondientes a las siguientes funciones:



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

def recortes(imgs, coords = None, indice = None):
    """Recibe una lista de imagenes y las coordenadas de los ROIs de las imagenes
    y devuelve una lista con las imagenes recortadas en las ROIs correspondientes"""
    if coords is None:
        #Calculo manual de coordenadas de patentes
        coords = coords = [(130, 180, 325, 425),#(x1,x2,y1,y2) con x eje vertical e y eje horizontal
                           (190, 240, 305, 405),
                           (170, 220, 225, 325),
                           (170, 220, 305, 405),
                           (105, 155, 240, 340),
                           (170, 220, 285, 385),
                           (150, 200, 235, 335),
                           (195, 245, 175, 275),
                           (220, 270, 355, 455),
                           (240, 290, 280, 380),
                           (190, 240, 240, 340),
                           (175, 225, 305, 405)] 
    if indice: #Para devolver 1 recorte
        i = indice - 1
        c = coords[i]
        x1, x2, y1, y2 = c[0], c[1], c[2], c[3]
        img = imgs[i]
        return img[x1:x2, y1:y2]
    recortes = []
    for i,c in enumerate(coords):
        x1, x2, y1, y2 = c[0], c[1], c[2], c[3]
        img = imgs[i]
        recortes.append(img[x1:x2, y1:y2])
    return recortes

def subplot12(imgs, titulo = "Auto", suptititle = "", recortar = False, coords = None, share = False):
    """Recibe 12 imagenes, 12 titulos y las muestra en un subplot identificando
       cada uno de los autos, del 1 al 12"""
    titles = []
    for i in range(1,13):
            titles.append(titulo + " " + str(i))
    if recortar:
        imgs = recortes(imgs)
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
    plt.suptitle(suptititle)
    plt.show(block=False)

def subplot4(img1, img2, img3, img4, t1 = "", t2 = "", t3 = "", t4=""):
    """Recibe una lista de imagenes y un indice y muestra 3 imagenes sobre
    el indice dado"""
    ax1 = plt.subplot(221); imshow(img1, new_fig=False, title=t1)
    plt.subplot(222, sharex=ax1, sharey=ax1); imshow(img2, new_fig=False, title=t2)
    ax2 = plt.subplot(223); imshow(img3, new_fig=False, title=t3)
    plt.subplot(224, sharex=ax2, sharey=ax2); imshow(img4, new_fig=False, title=t4)
    plt.show(block=False)

def muestra1(imgs1, imgs2, indice, t1 = "", t2 = ""):
    i = indice - 1
    img1 = imgs1[i]
    img2 = imgs2[i]
    patente1 = recortes(imgs1, indice=indice)
    patente2 = recortes(imgs2, indice=indice)
    t3 = " Patente " + t1
    t4 = " Patente " + t2
    subplot4(img1,img2, patente1, patente2,t1,t2,t3,t4)

def carga(viewAutos = False, viewPatentes = False, viewI = None):
    """Cargamos las imagenes, las almacenamos en una lista y la retornamos"""
    imgs = []
    for i in range(1,13):
        img = cv2.imread("img" + str(i).zfill(2) + ".png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    # Visualizamos el resultado
    if viewAutos:
        subplot12(imgs, suptititle="Original")
    if viewPatentes:
        subplot12(imgs, "Patente", "Original", recortar=True)
    if viewI:
        i = viewI - 1
        img = imgs[i]
        patente = recortes(imgs, indice=viewI)
        ax1 = plt.subplot(121); imshow(img, new_fig=False, title="Imagen original")
        plt.subplot(122); imshow(patente, new_fig=False, title="Patente original")
        plt.show()
    return imgs

def grises(imgs, viewAutos = False, viewPatentes = False, viewI = None):
    """Recibe una lista de imagenes, las convierte en escala de grises y las
    retorna en una lista"""
    grays = []
    for img in imgs:
        grays.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    # Visualizamos el resultado
    t1 = "Original"
    t2 = "Escala de grises"
    if viewAutos:
        subplot12(imgs,suptititle=t1)
        subplot12(grays,suptititle=t2)
    if viewPatentes:
        subplot12(imgs, "Patente",t1, recortar=True)
        subplot12(grays, "Patente",t2, recortar=True)
    if viewI:
        muestra1(imgs,grays,viewI, t1,t2)
    return grays

def highBoost(grays,a = 2, k = 5, viewAutos = False, viewPatentes = False, viewI = None):
    """Recibe una lista de imagenes en escala de grises y aplica un filtro high boost
    a todas las imagenes y devolvemos el resultado en una lista"""
    if len(grays[0].shape) != 2: #Si la imagen no esta en escala de grises imprime error y termina
        print("Error! tipo de imagen invalido, debe pasar una lista de imagenes en escala de grises")
        return -1
    grays_hb = []
    #Seteamos los parametros del filtro
    A =  a
    k2 = k**2
    pos = k // 2
    w = -np.ones((k,k))/k2
    w[2,2] = (k2*A-1)/k2
    for gray in grays:
        grays_hb.append(cv2.filter2D(gray,-1,w))
    # Visualizamos el resultado
    t1 = "Escala de grises"
    t2 = "Escala de grises + High boost"
    if viewAutos:
        subplot12(grays,suptititle=t1)
        subplot12(grays_hb,suptititle=t2)
    if viewPatentes:
        subplot12(grays,"Patente",t1, recortar=True)
        subplot12(grays_hb, "Patente",t2, recortar=True)
    if viewI:
        muestra1(grays,grays_hb,viewI, t1,t2)
    return grays_hb

def umbral(grays, th = 145, viewAutos = False, viewPatentes = False, viewI = None): # 145 Justo para detectar todos los caracteres sin que se rompan
    """Recibe una lista de imagenes en escala de grises y aplica un umbral dado
    a todas las imagenes y devolvemos el resultado en una lista"""
    if len(grays[0].shape) != 2: #Si la imagen no esta en escala de grises imprime error y termina
        print("Error! tipo de imagen invalido, debe pasar una lista de imagenes en escala de grises")
        return -1
    imgs_bin = []
    for gray in grays:
        _, img_bin = cv2.threshold(gray, th, 1, cv2.THRESH_BINARY)
        imgs_bin.append(img_bin)
    # Visualizamos el resultado
    t1 = "Escala de grises + High Boost"
    t2 = "Imagen binaria - Threshold > " + str(th)
    if viewAutos:
        subplot12(grays, suptititle=t1)
        subplot12(imgs_bin, suptititle=t2)
    if viewPatentes:
        subplot12(grays, "Patente", t1, recortar=True)
        subplot12(imgs_bin, "Patente", text, recortar=True)
    if viewI:
        muestra1(grays,imgs_bin,viewI, t1,t2)
    return imgs_bin

def filtro_area(imgs_bin,th1 = 18,th2 = 90, viewAutos = False, viewPatentes = False, viewI = None):
    imgs_bin_fa = [img.copy() for img in imgs_bin]
    for img in imgs_bin_fa:
        connectivity = 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)  
        for i in range(num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if (area < th1 or area > th2):
                img[labels==i] = 0
    # Visualizamos el resultado
    t1 = "Imagen binaria"
    t2 = "Filtrado " + str(th1) + " > Area > " + str(th2) 
    if viewAutos:
        subplot12(imgs_bin, suptititle=t1)
        subplot12(imgs_bin_fa, suptititle=t2)
    if viewPatentes:
        subplot12(imgs_bin, "Patente", t1, recortar=True)
        subplot12(imgs_bin_fa, "Patente", t2, recortar=True)
    if viewI:
        muestra1(imgs_bin,imgs_bin_fa,viewI, t1,t2)
    return imgs_bin_fa

def filtro_relacion_aspecto(imgs_bin,ra1 = 1.3,ra2 = 2.4, viewAutos = False, viewPatentes = False, viewI = None):
    imgs_bin_fra = [img.copy() for img in imgs_bin]
    for img in imgs_bin_fra:
        connectivity = 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)  
        for i in range(num_labels):
            alto = stats[i, cv2.CC_STAT_HEIGHT]
            ancho = stats[i, cv2.CC_STAT_WIDTH]
            relAsp = alto / ancho
            if (relAsp < ra1 or relAsp > ra2):
                img[labels==i] = 0
    # Visualizamos el resultado
    t1 = "Imagen binaria"
    t2 = str(ra1) + " > Rel. Asp > " + str(ra1)
    if viewAutos:
        subplot12(imgs_bin, suptititle=t1)
        subplot12(imgs_bin_fra, suptititle=t2)
    if viewPatentes:
        subplot12(imgs_bin, "Patente", t1, recortar=True)
        subplot12(imgs_bin_fra, "Patente", t2, recortar=True)
    if viewI:
        muestra1(imgs_bin,imgs_bin_fra,viewI, t1,t2)
    return imgs_bin_fra

# 7 # Calculamos nuevamentes las componentes conectadas y filtramos por grupos
#  de 3 componentes cercanas horizontalmente y verticalmente
def filtro_grupos3_cercanos(imgs_bin, viewAutos = False, viewPatentes = False, viewI = None):
    pass
    # imgs_th_fA_fRA_fG = [img.copy() for img in imgs_th_fA_fRA]
    # for img in imgs_th_fA_fRA_fG:
    #     connectivity = 8
    #     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    #     #Ordeno stats segun la coordenada izquierda de menor a mayor
    #     sorted_indices = np.argsort(stats[:, cv2.CC_STAT_LEFT])  # Índices que ordenan por la coordenada izquierda
    #     stats = stats[sorted_indices]  # Ordeno stats según la coordenada izquierda
    #     new_labels = np.zeros_like(labels)  # Creamos una nueva matriz de etiquetas del mismo tamaño
    #     for i, idx in enumerate(sorted_indices):
    #         new_labels[labels == idx] = i
    #     # Defino umbral de distancia
    #     dist_max_h = 13 #13 va bien, faltan 2 patentes que quedaron incompletas, INVESTIGAR POR SEPARADO EN DETALLE
    #     dist_max_v = 5
    #     # Lista para guardar los grupos de 3 componentes cercanos
    #     indices = []
    #     intentos = 3
    #     while len(indices) < 6 and intentos >= 0: #Si no obtengo 6 caracteres bajo el umbral de distancia horizontal durante 3 intentos
    #         print(dist_max_h)
    #         for i in range(num_labels - 2):  # Verificamos grupos de 3
    #             i2 = i + 1
    #             i3 = i + 2
    #             x1 = stats[i, cv2.CC_STAT_LEFT]
    #             x2 = stats[i2, cv2.CC_STAT_LEFT]
    #             x3 = stats[i3, cv2.CC_STAT_LEFT]
    #             y1 = stats[i, cv2.CC_STAT_TOP]
    #             y2 = stats[i2, cv2.CC_STAT_TOP]
    #             y3 = stats[i3, cv2.CC_STAT_TOP]
    #             distH_1_2 = x2 - x1 <= dist_max_h
    #             distH_2_3 = x3 - x2 <= dist_max_h
    #             distV_1_2 = abs(y2 - y1) <= dist_max_v  
    #             distV_2_3 = abs(y3 - y2) <= dist_max_v
    #             # Verificar si las distancias entre componentes consecutivos son menores al umbral
    #             if distH_1_2 and distH_2_3 and distV_1_2 and distV_2_3 and i not in indices:
    #                 indices += [i, i2, i3]
    #             if len(indices) == 6: #Si obtengo 2 grupos de 3 componentes(caracteres) termino
    #                 break
    #         dist_max_h -= 1
    #         intentos -= 1
    #     # Elimino las componentes que no correspondan a los indices de los grupos de 3
    #     for i in range(num_labels):
    #         if i not in indices:
    #             print(i)
    #             img[new_labels==i] = 0

    # # 7.1 # Visualizamos el resultado
    # titles_fA_fRA_fG = []
    # for i in range(1,13):
    #     titles_fA_fRA_fG.append("Auto " + str(i) + "- Filtro grupos 3 cercanos")

    # titlesP_fA_fRA_fG = []
    # for i in range(1,13):
    #     titlesP_fA_fRA_fG.append("Patente " + str(i) + "- Filtro grupos 3 cercanos")

    # # Autos
    # subplot12(imgs_th_fA_fRA,titles_fA_fRA)
    # subplot12(imgs_th_fA_fRA_fG,titles_fA_fRA_fG)

    # # Patentes
    # patentes1 = recortes(imgs_th_fA_fRA)
    # patentes2 = recortes(imgs_th_fA_fRA_fG)
    # subplot12(patentes1,titles_fA_fRA)
    # subplot12(patentes2,titlesP_fA_fRA_fG)

    # return !

# 8 # Calculamos nuevamente componentes conectadas, guardamos coordenadas y 
# creamos bounding box en imagen original
def boundings():
    pass
    # coords2 = []
    # k = 3
    # h = 1
    # r = 10

    # # Identificacion de patentes en imagen original
    # identificacion = [img.copy() for img in imgs]

    # # Identifiacion de caracteres de patentes binarios
    # caracteres = [img.copy() for img in imgs_th_fA_fRA_fG]
    # for i, img in enumerate(caracteres):
    #     caracteres[i] = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2RGB)


    # for i,img in enumerate(imgs_th_fA_fRA_fG):
    #     connectivity = 8
    #     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    #     sorted_indices = np.argsort(stats[:, cv2.CC_STAT_LEFT])  # Índices que ordenan por la coordenada izquierda
    #     stats = stats[sorted_indices]
    #     # Guardamos la coordenada del punto medio derecho de la primer letra 
    #     x1 = stats[1, cv2.CC_STAT_LEFT] - k
    #     y1 = stats[-1, cv2.CC_STAT_TOP] - k
    #     argmax = np.argmax(stats[:, cv2.CC_STAT_LEFT])
    #     x2 = stats[argmax, cv2.CC_STAT_LEFT] + stats[argmax, cv2.CC_STAT_WIDTH] + k
    #     argmax = np.argmax(stats[:, cv2.CC_STAT_TOP]) 
    #     y2 = stats[argmax, cv2.CC_STAT_TOP] + stats[argmax, cv2.CC_STAT_HEIGHT] + k
    #     # Resaltamos los caracteres con el high boost que hicimos anteriormente
    #     recorte_gris = grays_hb[i][y1:y2,x1:x2]
    #     recorte_rgb = cv2.cvtColor(recorte_gris, cv2.COLOR_GRAY2RGB)
    #     # Pegar el recorte en la imagen RGB en la posición deseada
    #     identificacion[i][y1:y2, x1:x2] = recorte_rgb
    #     cv2.rectangle(identificacion[i], (x1, y1), (x2, y2), color=(0,255,0), thickness=2)
    #     cv2.rectangle(caracteres[i], (x1, y1), (x2, y2), color=(255,255,255), thickness=1)
    #     # Calculo las componentes conectadas para obtener caracteres y hacer bounding box
    #     i_char = 1
    #     for st in stats[1:]:
    #         yc1, xc1, yc2, xc2 = st[1] - h, st[0] - h, st[1]+st[3] + h, st[0]+st[2] + h
    #         cv2.rectangle(identificacion[i], (xc1, yc1), (xc2, yc2), color=(255,0,0), thickness=1)
    #         pos = (xc1 + (xc2-xc1)//2 - 3, yc1-2)
    #         text = str(i_char)
    #         i_char += 1
    #     caracteres[i] = caracteres[i][y1-r:y2+r,x1-r:x2+r]
    #     coord = (y1-r,y2+r,x1-r,x2+r) #Agrando para obtener un recorte mas prolijo y automatico sin usar coords manual
    #     coords2.append(coord) # Invertido para usar con funcion recortes


    # subplot12(caracteres, titlesO)
    # patentes = recortes(identificacion, coords2) #Estas coordenadas son automaticas, no manuales como coord
    # subplot12(patentes, titlesO)

    # return !

def resultados_final():
    pass
    # for i, img in enumerate(imgs):
    #     plt.figure()
    #     plt.subplot(221); imshow(img, new_fig=False, title="Auto " + str(i))
    #     x1, x2, y1, y2 = coords2[i]
    #     plt.subplot(222); imshow(identificacion[i], new_fig=False, title="Deteccion patente " + str(i))
    #     plt.subplot(223); imshow(patentes[i], new_fig=False, title="Patente " + str(i))
    #     plt.subplot(224); imshow(caracteres[i], new_fig=False, title="Caracteres " + str(i))
    #     plt.show(block=False)


# Valor de imagen de auto a visualizar individualmente en cada invocacion
n = 1
imgs = carga(True, True, n)

grays = grises(imgs, True, True, n)

grays_hb = highBoost(grays, 2, 5, True, True, n)

imgs_bin1 = umbral(grays_hb, 145, True, True, n)

imgs_bin2 = filtro_area(imgs_bin1, 18, 90, True, True, n)

imgs_bin3 = filtro_relacion_aspecto(imgs_bin2, 1.3, 2.4, True, True, n)

imgs_bin4 = filtro_grupos3_cercanos(imgs_bin3, True, True, n)

identificacion = boundingsBox()

