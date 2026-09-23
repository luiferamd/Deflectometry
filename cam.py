from instrumental.drivers.cameras import uc480
import os
import time
import cv2
import matplotlib.pyplot as plt

SAVE_PATH= r"C:\Proyectos UN\TDG\Deflectometry\images"
CONFIG_PATH = r"C:\Proyectos UN\TDG\Deflectometry\deflectometry_settings.tcp"

class thorCam:
    def __init__(self, camera_index, save_path=SAVE_PATH):
        self.index= camera_index
        self.save_path= SAVE_PATH
        #detectar cámaras
        instruments= uc480.list_instruments()
        #Msj de error en caso de no haber conexión con la cámara
        if len(instruments) == 0:
            raise RuntimeError("No se encontraron cámaras UC480.")
        #Msj de error en caso de pasar un indice no disponible
        if camera_index >= len(instruments):
            raise IndexError(
                f"Indice de cámara inválido"
            )
        
        #instanciar cámara
        self.cam= uc480.UC480_Camera(instruments[camera_index])
       
        self.last_image = None
        self.display= None

    #Método para configurar cámara, principalmente recibirá una ruta con el archivo de config cargado desde ThorCam
    def configure(self,config=None, width=None, height=None, exposure=None, gain=None):

        if config is not None:
            #load_params cargará la configuración guardada en un .tcp de Thorcam, pasar PATH
            self.cam.load_params(config)
            print("Exposure:", self.cam.exposure)
            print("master_gain:", self.cam.master_gain)
            print("Width:", self.cam.width)
            print("Height:", self.cam.height)
            print("Gamma:", self.cam.gamma)
            print("Blacklevel:", self.cam.blacklevel_offset)
            print("Internal color mode:", self.cam._color_mode)
            print("Internal color depth:", self.cam._color_depth)

        
        if exposure is not None:
            self.cam.exposure= exposure
            print(f"Exposición configurada:{exposure}")
        if gain is not None:
            self.cam.gain= gain
            print(f"Ganancia configurada: {gain}")
        
    def capture(self):
        """
        Método para tomar fotografía con grab_image de uc480, devuelve en escala de grises
        """

        #grab_image devuelve array de numpy
        self.last_image = self.cam.grab_image(timeout= "1s", copy=True, exposure_time=self.cam.exposure)
        self.last_image = cv2.cvtColor(self.last_image, cv2.COLOR_RGB2GRAY)


        return self.last_image 
  
    def save_capture(self, name, image=None):
        """
        Método para guardar imagen en 
        INPUT: name -> str, nombre para reconocer imagen
        OUTPUT: full_path, opcional
        """
        # Si no se proporciona una imagen, utilizar la última captura
        if image is None:
            image = self.last_image

        if image is None:
            raise RuntimeError("No hay ninguna imagen para guardar.")
        
        #Datos de la carpeta de capturas según el día
        # Fecha actual para crear la carpeta: YYYY-MM-DD
        date_folder = time.strftime("%Y-%m-%d")
        # Ruta de la carpeta correspondiente al día
        save_folder = os.path.join(self.save_path, date_folder)
        # Crear la carpeta si no existe
        os.makedirs(save_folder, exist_ok=True)


        #Se toma dato de hora-min-seg para agregar al nombre de las imágenes
        #Se guardan los segundos para evitar que se sobreescriban imágenes en el mismo minuto
        timestamp = time.strftime("%H%M%S")
        filename = f"{name}_{timestamp}.tiff"
        full_path = os.path.join(save_folder, filename)
        success = cv2.imwrite(full_path, image)
    
        if not success:
            raise IOError(f"No se pudo guardar la imagen")

        print(f"Captura guardada en: {full_path}")
        """
        print("dtype:", self.last_image.dtype)
        print("shape:", self.last_image.shape)
        print("min:", self.last_image.min())
        print("max:", self.last_image.max())
        print("mean:", self.last_image.mean())
        
        """
        
        
        #Retornar full_path por si es necesario más adelante, se puede eliminar en futuro
        return full_path
    #Método para abrir visor en vivo
    def viewer(self):
        self.cam.start_live_video(exposure_time=self.cam.exposure)
        print("Exposure:", self.cam.exposure)

        print("Q: cerrar")
        print("S: capturar y guardar")

        while True:
            #Obtener frame
            self.capture()
            
            #Mostrar imagen
            #self.last_image = cv2.normalize(self.last_image, None, 0, 255, cv2.NORM_MINMAX)
            self.display = cv2.cvtColor(self.last_image, cv2.COLOR_RGB2BGR)

            cv2.imshow("Thorcam - Viewer", self.display)
            #Esperar 1ms por tecla
            key= cv2.waitKey(1) & 0xFF
            #Establecemos comandos con Q y S
            if key == ord("q"):
                break
            elif key == ord("s"):
                self.save_capture()

        cv2.destroyWindow("Thorcam - Viewer")

    def close(self):
        self.cam.close()
        print("Comunicación con la cámara cerrada")


if __name__ == "__main__":
    cam = thorCam(
        camera_index=0,
        save_path= SAVE_PATH
    )

    try:
        #configuración
        cam.configure(
            config= CONFIG_PATH
        )

        #abrir visor
        cam.viewer()
    finally:
        cam.close()
