"""
En este script se definen métodos de adquisición de imágenes de Deflectometría,
utilizando técnica de Phase-Shifting de 4 pasos para adquisición de información de la fase
-Se llama la clase de control de cámara desde cam.py
-Se llaman funciones de generación de patrones desde patterns.py

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from cam import thorCam #control de cámara
import patterns #generador de franjas
#unwrap_phase de skimage está pensada para imagenes 2D
from skimage.restoration import unwrap_phase


#Definición de parámetros para la cámara
CAMERA_INDEX = 0
SAVE_PATH= r"C:\Proyectos UN\TDG\Deflectometry\images"
CONFIG_PATH = r"C:\Proyectos UN\TDG\Deflectometry\deflectometry_settings.tcp" #Archivo con la config guardada desde ThorCam

#Definción de parámetros para los patrones sinosoidales
PERIOD = 40 #Periodo en pixeles
PHASE_STEPS = [-np.pi, #Phase shifting de 5 pasos
         -np.pi/2,
         0,   
         np.pi/2,
         np.pi
]
WINDOW_NAME = "Patron sinusoidal"
SCREEN_WIDTH= 1920
SCREEN_HEIGHT = 1080



def phi_wrapped(I1,I2,I3,I4,I5):
    """
    Input: I0, I1, I2, I3 -> Arrays de numpy en grayscale
    Output: phi_unwrapped -> Array float32
    phi_unwrapped recibe 4 capturas y con ellas hace el cálculo de la fase. 
    -Cálcula la fase con arctan2 usando la solución de 4 pasos del sistema de ecuaciones de Face Shifting, esta fase queda envuelta
    -Devuelve la fase envuelta en float32
    """
    # Convertimos a float para evitar problemas con las
    # operaciones aritméticas sobre imágenes uint8.
    I1 = I1.astype(np.float32)
    I2 = I2.astype(np.float32)
    I3 = I3.astype(np.float32)
    I4 = I4.astype(np.float32)
    I5 = I5.astype(np.float32)

    #Phase-shiffting de 4 pasos
    phi_wrapped = np.arctan2(
        2*(I2 - I4),
        ((2*I3)-I5-I1)
    )
    #normalizamos y cambiamos a uint8 para mostrar con opencv
    phi_wrapped_display = cv2.normalize(
                phi_wrapped,
                None,
                0,
                255,
                cv2.NORM_MINMAX)
    
    phi_wrapped_display = phi_wrapped_display.astype(np.uint8)
    cv2.imshow("Fase envuelta",phi_wrapped_display)

    return phi_wrapped

def phi_compensate(phi_wrapped, p=None):
    #factor de escala "p", tiene que ver conn el cambio de frecuencia debido a la propagación del patrón de franjas hacia la superficie
    if p is None:
        p=80.7 #obtenido previamente por iteraciones
    #mapa de pantalla
    x_s = np.arange (1280)
    y_s = np.arange (1024) 
    X , Y = np.meshgrid(x_s, y_s)
    #Usamos solo X ya que las franjas varían en el ancho
    fasor = np.exp(1j * -2*np.pi* (1/PERIOD)*X*p)
    phi_fasor = np.exp(1j * phi_wrapped)
    phi_compensated = fasor * phi_fasor
    phi_compensated = np.angle(phi_compensated)

    return phi_compensated

def amplitud_map(I1,I2,I3,I4,I5):
    # Convertimos a float para evitar problemas con las
    # operaciones aritméticas sobre imágenes uint8.
    I1 = I1.astype(np.float32)
    I2 = I2.astype(np.float32)
    I3 = I3.astype(np.float32)
    I4 = I4.astype(np.float32)
    I5 = I5.astype(np.float32)
    #Calcular mapa de amplitud
    amp_map= np.sqrt((2*(I2 - I4))**2 + ((2*I3)-I5-I1)**2)/2
    print("min amplitud:",amp_map.min())
    print("max amplitud:",amp_map.max())
    print("mean amplitud:",amp_map.mean())
    #normalizar para mostrar en opencv
    amp_map_display= cv2.normalize(amp_map,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX)
    amp_map_display= amp_map_display.astype(np.uint8)
    cv2.imshow("Mapa de amplitud",amp_map_display)


    return amp_map, amp_map_display


def mask_phase(phi, amp_map, threshold, kernel_size= None):
    """
    Genera una máscara binaria a partir del mapa de amplitud y la aplica sobre la fase envuelta. 
    La máscara identifica las regiones donde la amplitud de las franjas supera el umbral establecido. 
    Opcionalmente se puede aplicar una operación morfológica de closing para eliminar pequeños agujeros de la máscara.
    INPUT: phi -> fase envuelta (float32), amp_map -> mapa de amplitud (float32), threshold -> para definir máscara (int), kernel_size-> kernel de closing (int)
    OUTPUT: phi_masked -> Máscara enmascarada (float32), mask -> Máscara utilizada (uint8)
    """
    #Crear primera máscara
    mask_initial = (amp_map > threshold).astype(np.uint8)
    mask_blur = cv2.GaussianBlur(
        mask_initial.astype(np.float32),
        (17, 17),
        0)
    print("min mask_blur:",mask_blur.min())
    print("max mask_blur:",mask_blur.max())
    print("mean mask_blur:",mask_blur.mean())
    mask = (mask_blur > 0.5).astype(np.uint8)


    #Aplicar operación de closing para cerrar las zonas de fase dentro del plato, y evitar perder su información de fase
    if kernel_size is not None: 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        mask = cv2.morphologyEx( mask, cv2.MORPH_CLOSE, kernel )

    #Crear array de booleanos que establecen que zonas de phi son inválidas para computar, 
    #mask==0 es porque ma.array considera true como zonas invalidas, entonces invertimos la lógica
    phi_masked = np.ma.array(phi,mask= (mask==0))

    return phi_masked, mask

def get_captures(cam):
    """
    INPUT: cam -> objeto instanciado de la cámara
    OUTPUT: captures[I0,I1,I2,I3] -> Capturas con patrones de franjas desfasados en grayscale
    get_captures se encarga de:
    - Construir ventana y proyectar patrones de franjas sinosoidales con las fases definidas en PHASE_SETPS
    - Tomar capturas con la cámara previamente configurada
    """
    captures = []
    #Construir ventana de proyección
    patterns.windows_builder(
        WINDOW_NAME,
        patterns.SCREEN_WIDTH,
        patterns.SCREEN_HEIGHT
    )
    #ADQUISICIÓN

    i=1 #contador para nombre de imágenes
    for phase in PHASE_STEPS:
        #Generar patrón
        pattern= patterns.pattern_builder(
            PERIOD,
            phase,
            patterns.AMP,
            patterns.OFFSET
        )
         # Mostrar patrón en la pantalla de proyección
        cv2.imshow(WINDOW_NAME,pattern)
        cv2.waitKey(150)
        #Capturar imagen en escala de grises
        image = cam.capture()
        name = f"grano_esmaltado_I{i}"
        cam.save_capture(name)
        i+=1

        #Visualizar para verificar qué ve la cámara en la ultima imagen
        cv2.imshow("Imagen adquirida", image)
        print("tamaño imagen:", image.shape)
        cv2.waitKey(1)
        captures.append(image.copy())
        print(
            f"Captura realizada. "
            f"Phase = {phase:.4f} rad"
        )
    return captures[0], captures[1], captures[2], captures[3], captures[4]

def main():
    #Instanciar cámara
    cam = thorCam (camera_index=CAMERA_INDEX, save_path= SAVE_PATH)
    try:
        cam.configure(config=CONFIG_PATH)
        print(
            f"Exposición: "
            f"{cam.cam.exposure}"
        )
        print("Iniciando adquisición")
        I1, I2, I3, I4, I5 = get_captures(cam)
        print("Calculando fase")
        #calculamos la fase envuelta
        phi = phi_wrapped (I1, I2, I3, I4, I5)
        #Calculamos mapa de amplitud y la función se encarga de visualizar
        amp , display = amplitud_map(I1, I2, I3, I4, I5)

        #Guardar mapa de amplitud
        name_amp= f"amplitud_grano_esmaltado"
        cam.save_capture(name_amp, display)
            
        #Compensación de la fase lineal acoplada en phi_masked por la el patrón propagada
        phi_compensated = phi_compensate(phi)
        #Enmascaramos la fase con el mapa de amplitud
        phi_masked , mask = mask_phase(phi_compensated, amp, 3.5, 50)
        #Desenvolvemos la fase previamente enmascarada
        #Se hace uso de unwrap_phase de skimage para desenvolver la fase
        phase_unwrapped= unwrap_phase(phi_masked)
        mask_display = mask * 255
        cv2.imshow(f"Mascara usada", mask_display)

        #normalizar para visualizar fase desenvuelta
        #cv2.normalize es preferible para no eliminar valores menores que 0 como sí hace np.clip
        phi_unwrapped_display = cv2.normalize(
                    phase_unwrapped,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )
        phi_unwrapped_display = phi_unwrapped_display.astype( np.uint8)
        
        cv2.imshow(f"Fase desenvuelta", phi_unwrapped_display)
        #Guardamos la fase
        name_phase = f"fase_grano_esmaltado"
        cam.save_capture(name_phase, phi_unwrapped_display)


        print("\nPresione Q para cerrar.")
        while True:

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
    finally:

        # ----------------------------------------------------
        # Cerrar cámara y ventanas
        # ----------------------------------------------------

        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
