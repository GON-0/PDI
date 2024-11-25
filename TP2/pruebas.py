# HB ------------------------------------------------------------------------

n = 0
img = grays[n]
A = 2 #1.8
w1 = -np.ones((3,3))/(3*3)
w1[1,1] = (9*A-1)/9
w2 = -np.ones((5,5))/(5*5)  
w2[2,2] = (25*A-1)/25
img1 = cv2.filter2D(img,-1,w1)
img2 = cv2.filter2D(img,-1,w2)

plt.figure()
ax1 = plt.subplot(131); imshow(img, new_fig=False, title="")
plt.subplot(132, sharex=ax1, sharey=ax1); imshow(img1, new_fig=False, title="")
plt.subplot(133, sharex=ax1, sharey=ax1); imshow(img2, new_fig=False, title="")
plt.show(block=False)

# TH ----------------------------------------------------------------------

TH1 = 142 #125 esta bastante bien
imgs_th1 = []
TH2 = 145 
imgs_th2 = []

for gray in grays_hb:
    _, img_bin = cv2.threshold(gray, TH1, 1, cv2.THRESH_BINARY)
    imgs_th1.append(img_bin)
    _, img_bin = cv2.threshold(gray, TH2, 1, cv2.THRESH_BINARY)
    imgs_th2.append(img_bin)

titles = []
for i in range(1,13):
    titles.append("Auto " + str(i) + " - Threshold > " + str(TH1))

titles2 = []
for i in range(1,13):
    titles2.append("Auto " + str(i) + " - Threshold > " + str(TH2))

# imshow(imgs_th1[n])
patentes1 = recortes(imgs_th1)
patentes2 = recortes(imgs_th2)
subplot12(patentes1,titles)
subplot12(patentes2,titles2)



# 9 # Dilatamos las letras para hacerlas mas gruesas
patentes_dil = []
for img in imgs_th_fA_fRA_fG_dil:
    L = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L, L) )
    img = cv2.dilate(img, kernel, iterations=1)
    patentes_dil.append(img)


titlesPD = []
for i in range(1,13):
    titlesPD.append("Patente " + str(i) + " - Dilatado con circulo radio " + str(L))

patentes = recortes(patentes_dil)
subplot12(patentes,titlesPD)

# 8 # Filtrado inverso para detectar pixels negros de la imagen

TH2 = 15 # Justo para detectar todos los caracteres sin que se rompan
imgs_th_black = []

for gray in grays_hb:
    _, img_bin = cv2.threshold(gray, TH2, 1, cv2.THRESH_BINARY_INV)
    imgs_th_black.append(img_bin)


# 8.1 # Visualizamos el resultado

titlesTHB = []
for i in range(1,13):
    titlesTHB.append("Auto " + str(i) + " - Threshold < " + str(TH2))

subplot12(imgs_th_black,titlesTHB)
patentes = recortes(imgs_th_black)
subplot12(patentes,titlesTHB)


# combinamos resultados anteriores con una union
patentesT = []
for i in range(len(imgs_th_black)):
    mask1 = patentes_dil[i]
    mask2 = imgs_th_black[i]
    patentesT.append(cv2.bitwise_or(mask1, mask2))

patentes = recortes(patentesT)
subplot12(patentes,titlesTHB)










connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)

    sorted_indices = np.argsort(stats[:, cv2.CC_STAT_LEFT])  # Índices que ordenan por la coordenada izquierda
    stats = stats[sorted_indices]