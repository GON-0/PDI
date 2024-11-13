import cv2 
import numpy as np
import matplotlib.pyplot as plt

"""Funcion para simplificar codigo visualizaciones"""
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
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


"""Ejercicio 1
a) En la figura 1 se muestra un Sudoku (archivo sudoku.jpeg), se pretenden detectar las 20
líneas que delimitan las celdas a partir de la transformada de Hough 'cv2.HoughLines()' y
dibujarlas en la imagen.

b) Luego, con el resultado obtenido anteriormente, se debe calcular automáticamente el
número total de celdas vacías para determinar el porcentaje de avance del juego
(independientemente si está correcto o no).

c) Por último, las celdas vacías deberán ser pintadas de color gris. El programa mostrará la
imagen final con las líneas detectadas resaltadas, las celdas vacías en gris y el porcentaje
de avance del juego como título de la imagen.
"""

"""a)"""
#Cargo la imagen
f = cv2.imread('sudoku.jpeg')
f.dtype
f.shape
imshow(f)

#Pasamos a escala de grises para usar canny
gray = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
imshow(gray)

# #Aplicamos metodo canny a la imagen para obtener los bordes, NO SIRVE DA COMO RESULTADO 40 lineas
# edges = cv2.Canny(gray, 100, 150, apertureSize=3)
# imshow(edges)

#A mano, transformamos la imagen para obtener 20 lineas
edges = np.uint8((~gray > 30) * 255)

imshow(edges)
#Creamos una copia de la imagen original para dibujar las lineas
f_lines = f.copy()

#Aplicamos el metodo houghlines para obtener las lineas de la imagen
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=200) 

#Dibujamos las lineas sobre la copia recien creada
for i in range(0, len(lines)):
    #Paso de coordenadas polares a cartesianas
    rho = lines[i][0][0]
    theta = lines[i][0][1]        
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+500*(-b))
    y1=int(y0+500*(a))
    x2=int(x0-500*(-b))
    y2=int(y0-500*(a))
    #Si necesito ver los extremos de las lineas
    # cv2.circle(f_lines, (int(x1),int(y1)), 2, (255,0,0), 2) 
    # cv2.circle(f_lines, (int(x2),int(y2)), 2, (255,0,0), 2) 
    cv2.line(f_lines,(x1,y1),(x2,y2),(0,255,0),2)

#Mostramos el resultado
imshow(f_lines, ticks=True)

"""b)"""

f.shape #Es casi cuadrada, por lo tanto facilita el calculo de las coordenadas
lines[:,0,:]
#Me quedo solo con las coordenadas (distancia al origen) de las lineas
coord = np.uint(np.delete(lines[:,0,:], 1,axis=1).reshape(-1))
#Convierto a entero
coord = [int(i) for i in coord]
#Ordenamos la lista de lineas
coord.sort()
#Me quedo con las coordenadas con indice par ya que son similares
coord = coord[0::2]
#Resultado
coord

#Visualizo las primeras celdas de 2x2 
compl = np.uint8(~gray > 30)
imshow(compl)
k = 7#Sumo y resto 7 para evitar bordes
c1 = compl[coord[0]+k:coord[1]-k,coord[0]+k:coord[1]-k] 
c2 = compl[coord[0]+k:coord[1]-k,coord[1]+k:coord[2]-k]
c3 = compl[coord[1]+k:coord[2]-k,coord[0]+k:coord[1]-k]
c4 = compl[coord[1]+k:coord[2]-k,coord[1]+k:coord[2]-k]
plt.subplot(221)
plt.imshow(c1, cmap="gray",vmin=0,vmax=255)
plt.subplot(222)
plt.imshow(c2, cmap="gray",vmin=0,vmax=255)
plt.subplot(223)
plt.imshow(c3, cmap="gray",vmin=0,vmax=255)
plt.subplot(224)
plt.imshow(c4, cmap="gray",vmin=0,vmax=255)
plt.show(block=False)

#En la funcion para visualizar hecha por el profe no se bien en algunos casos
# imshow(c1)
# imshow(c2)
# imshow(c3)
# imshow(c4)

celdas = []
ubi_celdas = []
k = 7 #Valor para reducir el tamaño de las celdas y evitar bordes(centrarnos en el contenido)
largo = len(coord)-1 #Son 9 celdas (cantLineas(10) - 1)
for i in range(largo):
    for j in range(largo):
        #Calculo la ubicacion de cada celda 
        x1,x2,y1,y2 = (coord[i],coord[i+1],coord[j],coord[j+1])
        ubi = (x1,x2,y1,y2)
        #Calculo ubicacion modificada
        ubi_mod = (x1+k,x2-k,y1+k,y2-k) #Sumo y resto k para las variables 1 y 2 respestivamente
        #Obtengo la imagen a partir de la ubicacion y la imagen complemento (binaria)
        celda = compl[ubi_mod[0]:ubi_mod[1],ubi_mod[2]:ubi_mod[3]]
        #Guardo las imagenes de cada celda para luego usarlas en el apartado c
        ubi_celdas.append(ubi) 
        #Verifico si existe al menos un pixel de valor 1 (== celda ocupada)
        celdas.append(celda.any())

#Transformo a matriz para una mejor visualizacion
celda = np.array(celdas).reshape(largo,largo) #OJO ACA, nombre de la variable es celda en singular !

#Muestro el resultado
total = largo * largo #(cantidad de filas por cantidad de columnas, al ser las mismas es largo^2)
ocupadas = np.sum(celdas)
vacias = total - ocupadas
porcentaje = round(ocupadas * 100 / total,2)


"""c)"""

f_final = gray.copy()
for i in range(largo * largo):
    if not celdas[i]: #Si la celda no esta ocupada (==vacia)
        x1,x2,y1,y2 = ubi_celdas[i]
        x2 += 1 #Para detalles minimos de completado de celda en gris
        y2 += 4 #Hago las celdas un poco mas anchas, esto es porque la imagen al no ser completamente cuadrada no queda bien del todo(probar sin esta linea)
        f_final[x1:x2,y1:y2] = 127
        #print('celda',[])


#Muestro resultado final, comparando imagen origical y resultante
plt.subplot(121)
plt.imshow(gray, cmap="gray",vmin=0,vmax=255), plt.title('Imagen original'), plt.xticks([]), plt.yticks([])
plt.colorbar()
plt.subplot(122)
plt.imshow(f_final, cmap="gray",vmin=0,vmax=255), plt.title('Avance ' + str(porcentaje) + '%'), plt.xticks([]), plt.yticks([])
plt.colorbar()
plt.show(block = False)



















"""Ejercicio 2
    A partir del dataset de imágenes que se muestra en la figura 2 (archivos tateti_<id>.png), se
    requiere realizar un análisis automático mediante técnicas de PDI, sobre cada partida de
    TA-TE-TI.

    Implemente los ítems que se detallan a continuación, en cada una de las imágenes del
    dataset provisto y exponga cada resultado obtenido.

2.1 Carga de la imagen de entrada
    a) Cargar la imagen desde el archivo tateti_<id>.png, extraer su información básica y
mostrarla en una figura

2.2 Detección de bordes y figuras geométricas
    b) Elaborar un script que implemente las siguientes consignas de forma automática:
        i)  A partir de la imagen original, convertir la misma a escala de grises y obtener
            su representación binaria con alguna técnica de detección de bordes.
            Nota: Para la detección puede hacer uso del método cv2.Canny().

        ii) Con la imagen binaria y alguna técnica de detección de líneas, identifique
            las rectas horizontales y verticales que conforman las divisiones del tablero.
            Nota: Para la detección puede hacer uso del método cv2.HoughLines().

        iii)Mediante el uso de subplots, mostrar la imagen en escala de grises, la
            imagen binaria y la imagen original con la representación de cada recta
            detectada (trazos de color azul), en una única figura.

        iv) Recortar y etiquetar cada región del tablero mediante el uso de la información
            provista por las rectas detectadas. Muestre el resultado en una única figura,
            utilizando subplots y respetando la disposición original de las mismas según
            el tablero de TA-TE-TI.
        v)  Utilizar alguno de los métodos conocidos que permita determinar si la misma
            posee algún elemento o si se encuentra vacía.

        vi) En aquellas regiones que posean un objeto, identificar si el mismo es un
            círculo o una cruz.
            Nota: Para la identificación puede hacer uso del método cv2.HoughCircles().

        vii)Bajo la disposición original del tablero de TA-TE-TI, utilizar subplots para
            mostrar cada recorte. Asignar la etiqueta correspondiente (Cruz, Círculo o
            Vacío), como título de los mismos.
        
    c) Extra: En este juego, cada jugador debe colocar una figura (cruz o círculo) en algún
       espacio vacío con el objetivo de formar una línea (tres figuras) horizontal, vertical o
       diagonal. Si el tablero se completa y ningún jugador pudo formar alguna de ellas, el
       juego finaliza en un empate.

        i)  Implementar en el script, una función que permita determinar de forma
            automática, si la partida concluyó y en tal caso, qué figura fue la vencedora
            (círculos o cruces).

        ii) Mostrar la imagen original exponiendo el resultado de la partida en su título.
"""

"""a)Visualizacion e informacion basica de las imagenes"""

partis = []
gray_partis = []
for i in range(9):
    n = str(i+1)
    f = cv2.imread('tateti_' + n + '.png')
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    print('----------Tateti ' + n + '----------')
    print('tipo de dato: ', f.dtype)
    print('dimensiones: ', f.shape)
    print('pixel min: ', f.min())
    print('pixel max: ', f.max())
    partis.append(f)
    gray_partis.append(gray)

    #Visualizacion
    if i == 0:
        ax = plt.subplot(331)
        plt.imshow(gray, cmap="gray",vmin=0,vmax=255), plt.title('Tateti ' + n), plt.xticks([]), plt.yticks([])
    else:
        plt.subplot(331+i,sharex=ax,sharey=ax)
        plt.imshow(gray, cmap="gray",vmin=0,vmax=255), plt.title('Tateti ' + n), plt.xticks([]), plt.yticks([])

plt.show()


"""b)"""

"""i)Deteccion de bordes"""
#AQUI DEBO IR CAMBIANDO DE IMAGEN PARA IR PROBANDO TODAS
f = partis[3]
gray = gray_partis[3]
#Posible opcion:
# gray = cv2.GaussianBlur(gray, ksize=(7, 7), sigmaX=1.5) #No hace falta, no hay detalles de fondo que ignorar
edges = cv2.Canny(gray, 90, 150, apertureSize=3)

imshow(edges)

"""i)Deteccion de lineas"""
#Creamos una copia de la imagen original para dibujar las lineas
f_lines = f.copy()
#Aplicamos el metodo houghlines para obtener las lineas de la imagen
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=300) 

#Dibujamos las lineas sobre la copia recien creada
for i in range(0, len(lines)):
    #Paso de coordenadas polares a cartesianas
    rho = lines[i][0][0]
    theta = lines[i][0][1]        
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+500*(-b))
    y1=int(y0+500*(a))
    x2=int(x0-500*(-b))
    y2=int(y0-500*(a))
    #Si necesito ver los extremos de las lineas
    # cv2.circle(f_lines, (int(x1),int(y1)), 2, (255,0,0), 2) 
    # cv2.circle(f_lines, (int(x2),int(y2)), 2, (255,0,0), 2) 
    cv2.line(f_lines,(x1,y1),(x2,y2),(0,255,0),1)

imshow(f_lines)

#Mostramos el resultado
"""iii)"""
plt.subplot(131)
plt.imshow(gray, cmap="gray",vmin=0,vmax=255), plt.title('Escala de grises')
plt.subplot(132)
plt.imshow(edges, cmap="gray",vmin=0,vmax=255), plt.title('Imagen binaria')
plt.subplot(133)
plt.imshow(f_lines,vmin=0,vmax=255), plt.title('Lineas detectadas')
plt.show()

"""iv)"""

lines[:,0,:]
#Me quedo solo con las coordenadas (distancia al origen) de las lineas
coord = np.uint(np.delete(lines[:,0,:], 1,axis=1).reshape(-1))
#Convierto a entero
coord = [int(i) for i in coord]
#Ordenamos la lista de lineas
coord.sort()
#Me quedo con las coordenadas con indice par ya que son similares
coord = coord[0::2]
#Resultado
coordh = coord[1::2]
coordh.insert(0,30)
coordh.append(470)
coordv = coord[0::2] 
coordv.insert(0,30)
coordv.append(470)

k = 6 #Sumo y resto segun corresponda para evitar bordes
#Checkeo visualizando primeras 2x2 celdas
c11 = edges[coordh[0]+k:coordh[1]-k,coordv[0]+k:coordv[1]-k] 
c12 = edges[coordh[0]+k:coordh[1]-k, coordv[1]+k:coordv[2]-k] 
c21 = edges[coordh[1]+k:coordh[2]-k, coordv[0]+k:coordv[1]-k]
c22 = edges[coordh[1]+k:coordh[2]-k, coordv[1]+k:coordv[2]-k] 

plt.subplot(221)
plt.imshow(c11, cmap='gray')
plt.subplot(222)
plt.imshow(c12, cmap='gray')
plt.subplot(223)
plt.imshow(c21, cmap='gray')
plt.subplot(224)
plt.imshow(c22, cmap='gray')
plt.show()


#Obenemos la ubicacion y la imagen de cada celda
ubi = []
celdas = []
celdas_gray = []
largo = len(coord)-1
for i in range(largo):
    for j in range(largo):
        x1,x2,y1,y2 = (coordh[i]+k, coordh[i+1]-k, coordv[j]+k, coordv[j+1]-k)
        cbin = edges[x1:x2, y1:y2]
        cgray = gray[x1:x2, y1:y2]
        celdas.append(cbin)
        celdas_gray.append(cgray)
        ubi.append((x1,x2,y1,y2))

for i in range(largo*largo): #3x3 = 9 celdas
    n=i+1
    plt.subplot(331+i)
    plt.imshow(celdas_gray[i], cmap='gray'), plt.title('Celda ' + str(n))

plt.show()

"""v)"""

#Represento las jugadas de cada celda con numeros
#Donde:
#0 -> Vacia
#1 -> Circulo
#2 -> Cruz

#Estrategia:
#Creo una matriz con todos 2, asumiendo que tengo todas cruces
#Luego determino las celdas vacias y los circulos y queda lista

valores = np.full((3,3),2)

for i in range(largo):
    for j in range(largo):
        if not celdas[i*3+j].any(): #Lo veo en las celdas binarias
            valores[i,j] = 0


"""vi)"""
#Probamos con algunos ejemplos
c = celdas_gray[3]
#c = cv2.medianBlur(c,5) #No hace falta ya que no hay detalles de fondo
circles = cv2.HoughCircles(c,cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=40, minRadius=20, maxRadius=60)
circles
#Si encuentra un circulo devuelve el mismo, sino None

for i in range(largo):
    for j in range(largo):
        n = i*3+j
        celda = gray[ubi[n][0]:ubi[n][1],ubi[n][2]:ubi[n][3]]
        circles = cv2.HoughCircles(celda,cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=40, minRadius=0, maxRadius=70)
        if circles is not None:
            valores[i,j] = 1


#Resultado final
valores
imshow(f)

"""vii)"""
valores = valores.reshape(-1)
valores
for i in range(largo*largo): #3x3 = 9 celdas
    plt.subplot(331+i)
    if valores[i] == 0:
        titulo = 'Vacia'
    elif valores[i] == 1:
        titulo = 'Circulo'
    else:
        titulo = 'Cruz'
    plt.imshow(celdas_gray[i], cmap='gray'), plt.title(titulo)

plt.show()


"""c)"""

"""i)"""
valores = valores.reshape(3,3)
valores

#Funcion para determinar la cantidad de vacias, circulos y cruces del juego
def cantidad(valores):
    cant = [0,0,0] #[cantVacias. cantCirculos, cantCruces]
    for i in range(largo):
        for j in range(largo):
            v = valores[i,j]
            if v == 0:
                cant[0] += 1
            elif v == 1:
                cant[1] += 1

    cant[2] = largo*largo - cant[0] - cant[1]
    return cant


# #Pruebas
# #+++
# #---
# #---
# valores[0,0] == valores[0,1] == valores[0,2]

# #+--
# #+--
# #+--
# valores[0,0] == valores[1,0] == valores[2,0]

def termino(valores):
    cant = cantidad(valores)

    ganador = 0
    if cant[0] > 4: #Si la cantidad de vacias es mayor a 4, el juego no termina
        print('Partida NO concluida')
        return ganador
    
    v = valores

    #Asumo que hay un unico ganador, si habria 2 juegos de cada valor devolveria que gana el segundo
    for valor in range(1,3): #Juego de valor 1 o 2 segun si son circulos o cruces respectivamente
        juegoV = np.all(valor == v, axis = 0).any()
        juegoH = np.all(valor == v, axis = 1).any()
        diag1 = np.diag(v)
        diag2 = np.diag(np.flip(v))
        juegoD1 = diag1[0] == diag1[1] == diag1[2] == valor
        juegoD2 = diag2[0] == diag2[1] == diag2[2] == valor 

        if juegoV or juegoH or juegoD1 or juegoD2:
            ganador = valor

    if cant[0] == 0 and ganador == 0:
        print('Empate')
        return -1
    elif ganador == 0:
        print('Partida no concluida')
    return ganador
    
#Calculo el resultado y muestro
ganador = termino(valores)

titulo = ''
if ganador == -1:
    titulo = 'Empate'
elif ganador == 0:
    titulo = 'No termina'
elif ganador == 1:
    titulo = 'Ganan los circulos'
else:
    titulo = 'Ganan las cruces'

#Muestro resultado
imshow(f, title=titulo)

#Seguir probando con todas las imagenes !!! ya probe con 0 y 3