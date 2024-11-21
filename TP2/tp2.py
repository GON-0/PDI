import numpy as np
import matplotlib.pyplot as plt
import cv2

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

# Cargamos la imagen
img = cv2.imread('monedas.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imshow(img)

# Pasamos a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow(gray)

# Hacemos blur con filtro de mediana para eliminar ruido
gray_blur = cv2.medianBlur(gray,7)
imshow(gray_blur)

# Detectamos los circulos y los dibujamos en una nueva imagen binaria
circles = cv2.HoughCircles(gray_blur,cv2.HOUGH_GRADIENT, 1, 200, param1=50, param2=50, minRadius=130, maxRadius=200)  # Circulos chicos

circles = np.uint16(np.around(circles))
monedas_bin = np.zeros_like(gray)
for i in circles[0,:]:
    cv2.circle(monedas_bin, (i[0],i[1]), i[2], 1, -1)   # draw the outer circle
    cv2.circle(monedas_bin, (i[0],i[1]), 2, 1, -1)      # draw the center of the circle

# Visualizamos el resultado
plt.imshow(monedas_bin, cmap="gray")
plt.show()

# Dilatamos los circulos para que cubran cada moneda o podemos hacerlo luego de
# "tapar" las monedas
L = 40  # Probar con 10 - 30 - 70
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L, L) )
monedas_bin_dil = cv2.dilate(monedas_bin, kernel, iterations=1)

ax1 = plt.subplot(121); imshow(monedas_bin, new_fig=False, title="Monedas original")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(Fd, new_fig=False, title="Monedas dilatadas")
plt.show(block=False)

# Copiamos la imagen de grises y eliminamos las monedas pintando de negro o gris oscuro

gray_dados = gray_blur.copy()
gray_dados[monedas_bin_dil == 1] = 0
imshow(gray_dados)


# Definir el umbral
threshold = 180

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
Acl = cv2.morphologyEx(dados_bin, cv2.MORPH_CLOSE, kernel)
imshow(Acl)

# TENGO 2 MASCARAS;
    #MONEDAS : monedas_bin
    #DADOS : dados_bin