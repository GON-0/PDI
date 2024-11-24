# Importamos librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
--------------------------------------------------------------------------------
                    IDENTIFICACIÓN DE MONEDAS Y DADOS
--------------------------------------------------------------------------------

El presente trabajo se basa en el procesamiento de una imagen que contiene monedas
y dados para la clasificacion e identificacion de los mismos.

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

def monedas(img, informe = False, views = False):
    """Recibe una imagen de monedas, segmenta las mismas, calcula e informa el total
    y cuantas monedas de cada tipo hay. 
    Devuelve la mascara de la segmentacion y una imagen con la identificacion de cada moneda.
    Se puede especificar la visualizacion del informe y el procesamiento de la
    imagen por separado"""
    # 1 # Pasamos la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2 # Aplicamos blur con filtro de mediana para eliminar ruido
    gray_blur = cv2.medianBlur(gray,7)
    # 3 # Detectamos los circulos
    circles = cv2.HoughCircles(gray_blur,cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=50, minRadius=130, maxRadius=200)
    # 4 # Creamos imagenes sinteticas e inicializamos contadores
    circles = np.uint16(np.around(circles))
    mask_monedas = np.zeros_like(gray)
    tipos_monedas = np.zeros_like(img)
    cant_10c = 0
    cant_50c = 0
    cant_1p = 0
    identificacion = img.copy()
    # 5 # Con las monedas encontradas creamos 3 imagenes nuevas, la mascara (binaria), 
    # la clasificacion (RGB) y la identificacion (RGB)
    for c in circles[0,:]:
        # 5.1 # Dibujo circulo blanco en la mascara
        cv2.circle(mask_monedas, (c[0],c[1]), c[2], 1, -1)
        # 5.2 # Segun sea el radio dibujo un circulo con un color y escribo un texto determinado 
        radio = c[2]
        xc = c[1]
        yc = c[0]
        if radio > 170 and radio < 190: #50 centavos
            cv2.circle(tipos_monedas, (yc,xc), radio, (255,0,0), -1)
            cant_50c += 1
            text = "50 C"
        elif radio > 150 and radio < 170: #1 peso
            cv2.circle(tipos_monedas, (yc,xc), radio, (0,255,0), -1)
            cant_1p += 1
            text = "1 P"
        elif radio > 120 and radio < 150: #10 centavos
            cv2.circle(tipos_monedas, (yc,xc), radio, (0,0,255), -1)  
            cant_10c += 1
            text = "10 C"
        # 5.3 # Calculamos coordendas del ROI para el bounding box
        x1, y1, x2, y2 = yc - radio, xc - radio, yc + radio, xc + radio
        cv2.rectangle(identificacion, (x1, y1), (x2, y2), color=(255,0,0), thickness=3)
        # 5.4 # Escribo el texto correspondiente
        pos = (x1 + (x2-x1)//2 - 60, y1-30)
        cv2.putText(identificacion, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
    # 6 # Escribimos referencia de colores de la clasificacion con colores
    text50c = "50 Centavos"
    pos50c = (1000, 130)
    color50c = (255, 0, 0)  # Rojo
    text1p = "1 Peso"
    pos1p = (1730, 130)
    color1p = (0, 255, 0)  # Verde
    text10c = "10 Centavos"
    pos10c = (2200, 130)
    color10c = (0, 0, 255)  # Azul
    cv2.putText(tipos_monedas, text50c, pos50c, cv2.FONT_HERSHEY_SIMPLEX, 3, color50c, 5)
    cv2.putText(tipos_monedas, text1p, pos1p, cv2.FONT_HERSHEY_SIMPLEX, 3, color1p, 5)
    cv2.putText(tipos_monedas, text10c, pos10c, cv2.FONT_HERSHEY_SIMPLEX, 3, color10c, 5)
    # 7 # Si corresponde mostramos el informe obtenido
    if informe:
        print("-"*20 + "Informe" + "-"*20 + "\n")
        print("Cantidad de monedas : ", cant_10c + cant_50c + cant_1p)
        print("Cantidad de monedas de 10 centavos : ", cant_10c)
        print("Cantidad de monedas de 50 centavos : ", cant_50c)
        print("Cantidad de monedas de 1 peso : ", cant_1p)
        print("-"*47 + "\n")
    # 8 # Si corresponde mostramos las imagenes obtenidas en el procesamiento
    if views:
        ax1 = plt.subplot(221); imshow(img, new_fig=False, title="Imagen original")
        plt.subplot(222, sharex=ax1, sharey=ax1); imshow(gray_blur, new_fig=False, title="Imagen en escala de grises + blur")
        plt.subplot(223, sharex=ax1, sharey=ax1); imshow(mask_monedas, new_fig=False, title="Mascara binaria monedas")
        plt.subplot(224, sharex=ax1, sharey=ax1); imshow(tipos_monedas, new_fig=False, title="Clasificación de monedas")
        plt.show(block=False)
        imshow(identificacion, title="Identificación de las monedas")
    # 9 # Retornamos la mascara binaria de las monedas y la identificacion en la imagen original
    return mask_monedas, identificacion

def dados(img, mask_monedas, informe = False, views = False):
    """Recibe una imagen de dados, segmenta los mismos, calcula e informa la cantidad
    y el numero de la cara superior de cada dado.
    Devuelve una mascara de la segmentacion y una imagen con la identificacion de cada dado y el resultado
    Para esto utiliza una mascara de circulos para "borrar" las monedas de la imagen.
    Se puede especificar la visualizacion del informe y el procesamiento de la
    imagen por separado"""
    # 1 # Pasamos la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2 # Aplicamos blur con filtro de mediana para eliminar ruido
    gray_blur = cv2.medianBlur(gray,7)
    # 3 # Dilatamos los circulos de la mascara 
    L = 40 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L, L) )
    mask_monedas_dil = cv2.dilate(mask_monedas, kernel, iterations=1)
    # 4 # Tapamos las monedas de la imagen con los circulos dilatados
    gray_dados = gray_blur.copy()
    gray_dados[mask_monedas_dil == 1] = 0
    # 5 # Umbralamos la imagen con un threshold adecuado
    threshold = 185
    _, mask_dados = cv2.threshold(gray_dados, threshold, 1, cv2.THRESH_BINARY)
    # 6 # Obtenemos las componentes conectadas
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dados, connectivity, cv2.CV_32S)  
    # 7 # Eliminamos las componentes con area menor a un threshold
    AREA_TH = 4000
    cont = 0
    for i in range(num_labels):
        if stats[i, cv2.CC_STAT_AREA] < AREA_TH:
            cont += 1
            mask_dados[labels==i] = 0
    # 8 # Aplicamos clausura a la mascara para reconstruir las caras de los dados (Sin circulos)
    L = 80 # Con 70 esta bien pero pongo de mas ya que los circulos podrian ser mas grandes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L, L))
    mask_dados = cv2.morphologyEx(mask_dados, cv2.MORPH_CLOSE, kernel)
    # 9 # Obtenemos los circulos de los dados de la imagen en la escala de grises
    circles = cv2.HoughCircles(gray_blur,cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=50, minRadius=10, maxRadius=40)
    circles = np.uint16(np.around(circles))
    dados = mask_dados.copy()
    # 10 # Dibujamos los circulos en la mascara de dados para que queden bien definidos
    for c in circles[0,:]:
        cv2.circle(dados, (c[0],c[1]), c[2], 0, -1)
    # 11 # Obtenemos los dados con las componentes conectadas
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dados, connectivity, cv2.CV_32S)
    # 12 # Guardamos cantidad de dados y creamos imagenes nuevas
    cant_dados = num_labels - 1
    numeros = [] # para almacenar los numeros que salen en los dados
    img_color = cv2.cvtColor(dados * 255, cv2.COLOR_GRAY2RGB)
    identificacion = img.copy()
    # 13 # Con los dados encontrados creamos 2 imagenes nuevas una identificacion
         # en mascara binaria(RGB) y la identificacion en la imagen original(RGB)
    for st in stats[1:]:
        # 13.1 # Segmentamos cada dado
        x1, y1, x2, y2 = st[1], st[0], st[1]+st[3], st[0]+st[2]
        dado = gray_blur[x1:x2,y1:y2]
        # 13.2 # Detectamos circulos de cada dado en la imagen en escala de grises y guardamos la cantidad
        circles = cv2.HoughCircles(dado,cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=50, minRadius=10, maxRadius=40)
        numero = circles.shape[1]
        numeros.append(numero)
        # 13.3 # Agregamos los bounding box de cada dado en las dos imagenes
        cv2.rectangle(img_color, (y1, x1), (y2, x2), color=(0,255,0), thickness=3)
        cv2.rectangle(identificacion, (y1, x1), (y2, x2), color=(0,255,0), thickness=3)
        # 13.4 # Agregamos el texto con el numero del dado encima de la bounding box
        pos = (y1 + (y2-y1)//2 - 25, x1-30)
        cv2.putText(img_color, str(numero), pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)
        cv2.putText(identificacion, str(numero), pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5)
    # 14 # Obtenemos el resultado de los dados (la suma del numero de cada dado)
         #  y escribimos el resultado en la imagen
    suma = sum(numeros)
    cv2.putText(img_color, "Resultado : " + str(suma), (1650, 130), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 5) # Opcional escribir el resultado en la imagen
    # 15 # Si corresponde mostramos las imagenes obtenidas en el procesamiento
    if informe:
        print("-"*20 + "Informe" + "-"*20 + "\n")
        print("Cantidad de dados : ", cant_dados)
        print("Suma de todos los dados : ", suma)
        print("-"*47 + "\n")
    # 16 # Retornamos la mascara binaria de las monedas y la identificacion en la imagen original
    if views:
        ax1 = plt.subplot(221); imshow(img, new_fig=False, title="Imagen original")
        plt.subplot(222, sharex=ax1, sharey=ax1); imshow(gray_blur, new_fig=False, title="Imagen en escala de grises + blur")
        plt.subplot(223, sharex=ax1, sharey=ax1); imshow(mask_dados, new_fig=False, title="Mascara binaria cara superior dados")
        plt.subplot(224, sharex=ax1, sharey=ax1); imshow(img_color, new_fig=False, title="Identificación de dados")
        plt.show(block=False)
        imshow(identificacion, title="Identificación de los dados - Resultado :" + str(suma))
    return mask_dados, identificacion

# Cargamos la imagen y la visualizamos
img = cv2.imread('monedas.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imshow(img)

mask_monedas, id_monedas = monedas(img, informe = True, views = True)

mask_dados, id_dados = dados(img, mask_monedas, informe = True, views = True)

imagen_combinada = np.maximum(id_monedas, id_dados)

# Mostramos la identificacion de cada objeto por separado
ax1 = plt.subplot(121); imshow(id_monedas, new_fig=False, title="Identificación de monedas")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(id_dados, new_fig=False, title="Identificación de dados")
plt.show(block=False)


# Calculamos la diferencia entre la imagen original y cada imagen con bounding boxes
diferencia_monedas = cv2.absdiff(img, id_monedas)
diferencia_dados = cv2.absdiff(img, id_dados)

# Crear máscaras binarias para las áreas de los bounding boxes
mascara_monedas = np.any(diferencia_monedas > 0, axis=2)
mascara_dados = np.any(diferencia_dados > 0, axis=2)

# Crear la imagen combinada basada en la imagen original
identificacion = img.copy()

# Aplicar colores definidos a los bounding boxes
identificacion[mascara_dados] = [255, 0, 0]
identificacion[mascara_monedas] = [0, 255, 0]

# Mostramos el resultado final con la identificacion de monedas y dados 
ax1 = plt.subplot(221); imshow(img, new_fig=False, title="Imagen original")
plt.subplot(222, sharex=ax1, sharey=ax1); imshow(mascara_monedas, new_fig=False, title="Mascara bounding box monedas")
plt.subplot(223, sharex=ax1, sharey=ax1); imshow(mascara_dados, new_fig=False, title="Mascara bounding box dados")
plt.subplot(224, sharex=ax1, sharey=ax1); imshow(identificacion, new_fig=False, title="Identificacion final de monedas y dados")
plt.show(block=False)

