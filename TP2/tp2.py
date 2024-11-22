# Importamos librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Definimos funcion para simplificar codigo de visualizaciones
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
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
    # 5 # Dibujamos los circulos en 2 imagenes nuevas, la mascara (binaria) y la clasificacion (RGB)
    for c in circles[0,:]:
        cv2.circle(mask_monedas, (c[0],c[1]), c[2], 1, -1)
        if c[2] > 170 and c[2] < 190: #50 centavos
            cv2.circle(tipos_monedas, (c[0],c[1]), c[2], (255,0,0), -1)
            cant_50c += 1
        elif c[2] > 150 and c[2] < 170: #1 peso
            cv2.circle(tipos_monedas, (c[0],c[1]), c[2], (0,255,0), -1)
            cant_1p += 1  
        elif c[2] > 120 and c[2] < 150: #10 centavos
            cv2.circle(tipos_monedas, (c[0],c[1]), c[2], (0,0,255), -1)  
            cant_10c += 1
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
    # 9 # Retornamos la mascara binaria de las monedas
    #HACER IMAGEN ORIGINAL CON LA IDENTIFICACION DE LAS MONEDAS(bounding box + cartel)
    return mask_monedas

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
    cant_dados = num_labels - 1
    numeros = []
    img_color = cv2.cvtColor(dados * 255, cv2.COLOR_GRAY2RGB)
    for st in stats[1:]:
        # Segmentamos cada dado
        x1, y1, x2, y2 = st[1], st[0], st[1]+st[3], st[0]+st[2]
        # Detectamos circulos de cada dado en escala de grises
        circles = cv2.HoughCircles(gray_blur[x1:x2,y1:y2],cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=50, minRadius=10, maxRadius=40)
        numero = circles.shape[1]
        numeros.append(numero)
        cv2.rectangle(img_color, (y1, x1), (y2, x2), color=(255,0,0), thickness=3)
        pos = (y1 + (y2-y1)//2 - 25, x1-30)
        cv2.putText(img_color, str(numero), pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 5)
    suma = sum(numeros)
    # cv2.putText(img_color, "Resultado : " + str(suma), (1650, 130), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 5) # Opcional escribir el resultado en la imagen
    if informe:
        print("-"*20 + "Informe" + "-"*20 + "\n")
        print("Cantidad de dados : ", cant_dados)
        print("Suma de todos los dados : ", suma)
        print("-"*47 + "\n")
    if views:
        ax1 = plt.subplot(221); imshow(img, new_fig=False, title="Imagen original")
        plt.subplot(222, sharex=ax1, sharey=ax1); imshow(gray_blur, new_fig=False, title="Imagen en escala de grises + blur")
        plt.subplot(223, sharex=ax1, sharey=ax1); imshow(mask_dados, new_fig=False, title="Mascara binaria cara superior dados")
        plt.subplot(224, sharex=ax1, sharey=ax1); imshow(img_color, new_fig=False, title="Identificación de dados - Resultado : ", str(suma))
        plt.show(block=False)
    #HACER IMAGEN ORIGINAL A COLOR CON LA IDENTIFICACION DE LOS DADOS (bounding box + cartel)
    return mask_dados

# Cargamos la imagen y la visualizamos
img = cv2.imread('monedas.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imshow(img)

mask_monedas = monedas(img, informe = True, views = True)

mask_dados = dados(img, mask_monedas, informe = True, views = True)












# Dilatamos los circulos para que cubran cada moneda o podemos hacerlo luego de
# "tapar" las monedas
L = 40  # Probar con 10 - 30 - 70
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L, L) )
monedas_bin_dil = cv2.dilate(monedas_bin, kernel, iterations=1)

ax1 = plt.subplot(121); imshow(monedas_bin, new_fig=False, title="Monedas original")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(monedas_bin_dil, new_fig=False, title="Monedas dilatadas")
plt.show(block=False)

# Copiamos la imagen de grises y eliminamos las monedas pintando de negro o gris oscuro

gray_dados = gray_blur.copy()
gray_dados[monedas_bin_dil == 1] = 0
imshow(gray_dados)


# Definir el umbral
threshold = 185

# Aplicar la función threshold
_, dados_bin = cv2.threshold(gray_dados, threshold, 1, cv2.THRESH_BINARY)
imshow(dados_bin)

# Calculo las componentes conectadas y elimino las de area menor a un threshold
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dados_bin, connectivity, cv2.CV_32S)  

AREA_TH = 4000
cont = 0
for i in range(num_labels):
    if stats[i, cv2.CC_STAT_AREA] < AREA_TH:
        cont += 1
        dados_bin[labels==i] = 0

imshow(dados_bin)


# Dilatamos los dados
L = 80 # Con 70 esta bien pero pongo de mas ya que los circulos podrian ser mas grandes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L, L))
dados_bin = cv2.morphologyEx(dados_bin, cv2.MORPH_CLOSE, kernel)
#dados_bin2 = cv2.morphologyEx(dados_bin2, cv2.MORPH_OPEN, kernel) # Queda redondeado, no esta mal, pero no hace falta tanto detalle
imshow(dados_bin)

# Pensé 2 alternativas: 
    #+ dejar los circulos de los dados en la mascara y reconstruirla
    # con dilatacion, esto hace que se achiquen los circulos, pero se mantienen visibles

    # Elimnar los circulos de los dados, se obtienen cuadrados, aplicar la mascara
    # a la imagen original en escala de grises y obtener los circulos mas definidos

# OPCION 1
# Obtengo los circulos pequeños de la imagen en escala de grises y los dibujo en la imagen binaria
# de esta forma obtengo circulos mas definidos que los obtenidos con la umbralizacion
circles = cv2.HoughCircles(gray_blur,cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=50, minRadius=10, maxRadius=40)  # Circulos chicos
circles = np.uint16(np.around(circles))

for c in circles[0,:]:
    cv2.circle(dados_bin, (c[0],c[1]), c[2], 0, -1)

imshow(dados_bin)

connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dados_bin, connectivity, cv2.CV_32S)  
imshow(labels)

# Analizar por separado y calcular el numero de cada dado, mostrar el dado
# y el titulo con el numero del dado. Utilizar houghcircle para cada recorte de imagen
# con un dado, con eso ya resuelvo todo, tengo todo listo.

# Creatividad. Crear !!!
# Agregar mas comentarios, dividir en funciones cada modulo o respuesta del enunciado
# agregar parametros para activar o desactivar la visualizacion de las imagenes paso a paso
# Hacer imagenes piolas para mostrar la clasificacion de monedas coloreadas y dados
# con el titulo con el valor del mismo



# # Coloreamos los elementos
#     labels = np.uint8(255/num_labels*labels)
#     # imshow(img=labels)
#     im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
#     for centroid in centroids:
#         cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
#     for st in stats:
#         cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=2)