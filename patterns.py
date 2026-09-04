import numpy as np
import cv2

#=====================
#Declaramos parámetros
#=====================
#Parámetros del patrón de franjas sinosoidal
PERIOD = 40 #Periodo en pixeles
PHASE = 0 #Fase en radianes
AMP = 127.5 #Amplitud
OFFSET = 127.5 #Intensidad media

#Dimensiones laptop (PC principal)
MAIN_WIDTH= 1920
MAIN_HEIGHT = 1080
#Dimensiones pantalla del montaje
SCREEN_WIDTH= 1920
SCREEN_HEIGHT = 1080
#====================================

"""
windows_builder construye y otorga posición a una ventana de OpenCV
Sus parámetros son: 
-name : str , nombre de ventana
-width : int, ancho de ventana
-height : int, alto de ventana
"""
def windows_builder (name, width, height):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    #dimensiones de ventana
    cv2.resizeWindow(name, width, height)
    #posiciones iniciales de la segunda pantalla
    x_i= MAIN_WIDTH
    y_i= 0
    #mover ventana a la segunda pantalla
    cv2.moveWindow(name,x_i, y_i)
    cv2.setWindowProperty(
        name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

"""
pattern_builder construye el patrón de franjas sinosidal y lo proyecta
siguiendo la ecuación: I(x) = offset + amplitude * cos(2*pi*x/period + phase)

Parámetros:

windows_name: str, nombre de la ventana previamente construida
period : int, periodo en pixeles
phase: int, fase en radianes 
amplitud: amplitud max del seno
offset: intensidad media
"""
def pattern_builder (period, phase, amplitude, offset):
    #Coordenadas horizontales
    x = np.arange (SCREEN_WIDTH)
    #Patron sinosidal 1D
    pattern_x= offset + (amplitude* np.cos(((2 * np.pi * x )/ period)+ phase))
    #replicar en eje y
    pattern_2d = np.tile(pattern_x, (SCREEN_HEIGHT, 1) )
    #limitar valores al rango válido de imágenes por seguridad, aunque 0<I(x)<255 
    pattern_2d = np.clip(pattern_2d, 0, 255)
    pattern= pattern_2d.astype(np.uint8)

    return pattern

if __name__ == "__main__": 
    windows_name = "Patrón sinosoidal"

    windows_builder(
        windows_name,
        SCREEN_WIDTH,
        SCREEN_HEIGHT
    )

    pattern= pattern_builder(
        PERIOD,
        PHASE,
        AMP,
        OFFSET
    )
    #Mostrar patrón
    cv2.imshow(windows_name, pattern)
    print("Patrón sinosidal proyectado")
    print("Q: cerrar")

    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyWindow(windows_name)