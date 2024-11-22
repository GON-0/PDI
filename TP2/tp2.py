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
    y cuantas monedas de cada tipo hay y devuelve una mascara de la segmentacion.
    Se puede especificar la visualizacion del procesamiento de la imagen paso a paso"""
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
    text50c = "50 centavos"
    pos50c = (1000, 130)
    color50c = (255, 0, 0)  # Rojo
    text1p = "1 peso"
    pos1p = (1730, 130)
    color1p = (0, 255, 0)  # Verde
    text10c = "10 centavos"
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
    return mask_monedas

def dados(img, mask_circles ,views = False):
    """Recibe una imagen de dados, segmenta los mismos, calcula e informa la cantidad
    y el numero de la cara superior de cada dado y devuelve una mascara de la segmentacion.
    Se puede especificar la visualizacion del procesamiento de la imagen paso a paso"""
    pass


# Cargamos la imagen y la visualizamos
img = cv2.imread('monedas.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imshow(img)

monedas(img, informe = True, views = True)












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