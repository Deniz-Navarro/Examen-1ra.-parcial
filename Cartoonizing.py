import cv2 as cv
import numpy as np

#Funcion para aplicar el filtro cartoon a una imagen
def cartoonize_image (img, ds_factor=4, sketch_mode=False):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Cambiamos la imagen a escala de grises
    img_gray = cv.medianBlur(img_gray, 7) #Aplicamos el median blur
    
    #Detectamos los bordes
    edges = cv.Laplacian(img_gray, cv.CV_8U, ksize=5)
    ret, mask = cv.threshold(edges, 100, 255, cv.THRESH_BINARY_INV)

    if sketch_mode:
        return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    img_small = cv.resize(img, None, fx = 1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv.INTER_AREA)
    num_repetitions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5

    for i in range(num_repetitions):
        img_small = cv.bilateralFilter(img_small, size, sigma_color, sigma_space)
    
    img_output = cv.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv.INTER_LINEAR)
    dst = np.zeros(img_gray.shape)

    dst = cv.bitwise_and(img_output, img_output, mask=mask)
    return dst

if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    cur_char = -1
    prev_char = -1
    aux = ''

    while True:
        ret, frame = cap.read()
        #cv.imshow('frame', frame)
        frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

        c = cv.waitKey(1)

        if c == 27:
            break
        if c > -1 and c != prev_char:
            cur_char = c
        prev_char = c
        if cur_char == ord('s'):
            cv.imshow('Cartoonize', cartoonize_image(frame, sketch_mode=True))
            aux = 's'
        elif cur_char == ord('c'):
            cv.imshow('Cartoonize', cartoonize_image(frame, sketch_mode=False))
            aux = 'c'
        #Aregramos una condicional para la tecla para tomar fotos, En este caso es con la X
        elif cur_char == ord('x'):
            if(aux == 's'):
                cv.imwrite('Cartoonize.png', cartoonize_image(frame, sketch_mode=True))
                aux = ''
                cur_char = 115
            elif(aux == 'c'):
                cv.imwrite('Cartoonize2.png', cartoonize_image(frame, sketch_mode=False))
                aux = ''
                cur_char = 99
            else:
                cv.imwrite('Normal.png', frame)
                cur_char = -1
        else:
            cv.imshow('Cartoonize', frame)
            aux = ''
        
    cap.release()
    cv.destroyAllWindows()

