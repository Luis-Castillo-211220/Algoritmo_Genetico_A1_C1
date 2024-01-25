import numpy as np
import random
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
import os
import shutil 
import cv2

def calcular_n(rango, resolucion_inicial, num_puntos):
    n = 0
    while not (2**(n-1) <= num_puntos <= 2**n):
        n += 1
    return n

def generar_individuo(n):
    return [random.randint(0, 1) for _ in range(n)]

def inicializar_poblacion(poblacion_inicial, n):
    return [generar_individuo(n) for _ in range(poblacion_inicial)]

def podar_poblacion(poblacion, poblacion_maxima, a, resolucion_nueva, maximizar):
    valores_x_poblacion = [calcular_valor_x(individuo, a, resolucion_nueva) for individuo in poblacion]
    valores_fx_poblacion = [calcular_fx(x) for x in valores_x_poblacion]
    mejor_valor, _, _ = calcular_estadisticas(valores_fx_poblacion, maximizar)

    indice_mejor = valores_fx_poblacion.index(mejor_valor)
    mejor_individuo = poblacion[indice_mejor]

    del poblacion[indice_mejor]
    
    while len(poblacion) > poblacion_maxima:
        indice_eliminar = random.randint(0, len(poblacion) - 1)
        del poblacion[indice_eliminar]

    poblacion.append(mejor_individuo)

    return poblacion

def seleccion_aleatoria(poblacion, prob_cruza):
    parejas = []
    for _ in range(len(poblacion) // 2):
        if random.random() < prob_cruza:
            parejas.append((random.choice(poblacion), random.choice(poblacion)))
    return parejas

def cruza_un_punto(individuo1, individuo2):
    punto_cruza = random.randint(1, len(individuo1) - 1)
    hijo1 = individuo1[:punto_cruza] + individuo2[punto_cruza:]
    hijo2 = individuo2[:punto_cruza] + individuo1[punto_cruza:]
    
    print(f"Cruza entre {individuo1} y {individuo2}")
    print(f"Punto de cruza: {punto_cruza}")
    print(f"Hijo 1: {hijo1}", "Valor entero: ", binario_a_entero(hijo1))
    print(f"Hijo 2: {hijo2}", "Valor entero: ", binario_a_entero(hijo2))
    
    return hijo1, hijo2

def mutacion(individuo, prob_mutacion_individuo, prob_mutacion_gen):
    if random.random() < prob_mutacion_individuo:
        print("El hijo si mutara")
        individuo_mutado = individuo[:]
        for i in range(len(individuo_mutado)):
            if random.random() < prob_mutacion_gen:
                individuo_mutado[i] = 1 if individuo_mutado[i] == 0 else 0
        return individuo_mutado
    else:
        print("El hijo no mutara")
        return individuo

def binario_a_entero(binario):
    return sum(val * (2**idx) for idx, val in enumerate(reversed(binario)))

def calcular_valor_x(individuo, a, resolucion_nueva):
    valor_entero = binario_a_entero(individuo)
    return a + valor_entero * resolucion_nueva

def calcular_fx(x):
    # Ejemplos de distintas funciones objetivo
    # return ((x**3 * math.sin(x)) / 100) + x**2 * math.cos(x)
    # return ((x**2 * math.cos(5*x))) - 3*x
    return ((x * math.cos(x)) * (math.sin(2*x))) + 2*x
    # return (3*x * math.cos(x) * math.sin(x)) * (math.log(abs(x)+0.1))
    # return ((x**2) * math.cos(x) * math.sin(x)) * (math.log(abs(x**3)+0.1))
    # return (math.sqrt(x**3) * (math.cos(x)))
    # return math.sqrt(abs(x**3)) * math.cos(x)
    # return (3*(x**7/3)) * math.cos(x) * math.sin(x)

def calcular_estadisticas(fitnesses, maximizar):
    if(maximizar):
        mejor = max(fitnesses)
        peor = min(fitnesses)
        promedio = sum(fitnesses) / len(fitnesses)
    else:
        mejor = min(fitnesses)
        peor = max(fitnesses)
        promedio = sum(fitnesses) / len(fitnesses)

    return mejor, peor, promedio

def crear_video(carpeta_imagenes, fps=1, nombre_video='video_generaciones.mp4'):
    imagenes = [img for img in os.listdir(carpeta_imagenes) if img.endswith(".png")]
    imagenes.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Ordenar las imágenes por número de generación

    frame = cv2.imread(os.path.join(carpeta_imagenes, imagenes[0]))
    altura, anchura, capas = frame.shape

    video = cv2.VideoWriter(nombre_video, 0, fps, (anchura, altura))

    for imagen in imagenes:
        video.write(cv2.imread(os.path.join(carpeta_imagenes, imagen)))

    cv2.destroyAllWindows()
    video.release()

def iniciar_algoritmo():
    try:
        a = float(entrada_a.get())
        b = float(entrada_b.get())
        resolucion_inicial = float(resolucion_inicialE.get())
        poblacion_inicial = int(poblacion_inicialE.get())
        poblacion_maxima = int(poblacion_maximaE.get())
        num_generaciones = int(generaciones_ent.get())
        prob_cruza = float(prob_cruzaE.get())
        prob_mutacion_individuo = float(prob_ind.get())
        prob_mutacion_gen = float(prob_gen.get())
        maximizar = opcion_maximizar.get()
        maximizar = maximizar == 1
        
        rango = b - a
        num_puntos = (rango / resolucion_inicial) + 1
        n = calcular_n(rango, resolucion_inicial, num_puntos)
        resolucion_nueva = rango / (2**n - 1)
        num_puntos_nueva = (rango / resolucion_nueva) + 1
        print("Cantidad total de bits a usar: ", n)
        print(f"resolucion inicial: {resolucion_inicial:.4f}, Num saltos inicial: {num_puntos:.4f}")
        print(f"Nueva resolucion: {resolucion_nueva:.4f}, Saltos nuevos: {num_puntos_nueva:.4f}")
        
        label_bits['text'] = f"Cantidad total de bits a usar: {n}"
        label_resolucion['text'] = f"Nueva resolucion: {resolucion_nueva:.4f}"
        # label_saltos['text'] = f"Saltos nuevos: {num_puntos_nueva:.4f}"
        label_saltos['text'] = f"Saltos nuevos: {num_puntos_nueva}"
    except ValueError:
        tk.messagebox.showerror("Error", "Por favor, ingrese valores validos.")

    return a, b, resolucion_inicial, poblacion_inicial, poblacion_maxima, num_generaciones, prob_cruza, prob_mutacion_individuo, prob_mutacion_gen, maximizar, n, resolucion_nueva

def algoritmo_genetico_con_seleccion_cruza():
    a, b, resolucion_inicial, poblacion_inicial, poblacion_maxima, num_generaciones, prob_cruza, prob_mutacion_individuo, prob_mutacion_gen, maximizar, n, resolucion_nueva = iniciar_algoritmo()
    poblacion = inicializar_poblacion(poblacion_inicial, n)
    mejores, peores, promedios = [], [], []
    tamanos_poblacion = []
    tamanos_poblacion.append(len(poblacion))
    
    carpeta_imagenes = "imagenes"
    if os.path.exists(carpeta_imagenes):
        shutil.rmtree(carpeta_imagenes)
    os.makedirs(carpeta_imagenes)

    puntos_x = np.linspace(a, b, 500)
    puntos_fx = np.array([calcular_fx(x) for x in puntos_x])
    
    for generacion in range(num_generaciones):
        print(f"\nGeneracion {generacion + 1}:")
        print(f"Poblacion antes de la cruza: {poblacion}")
        parejas = seleccion_aleatoria(poblacion, prob_cruza)
        
        for pareja in parejas:
            hijo1, hijo2 = cruza_un_punto(pareja[0], pareja[1])
            hijo1_mutado = mutacion(hijo1, prob_mutacion_individuo, prob_mutacion_gen)
            hijo2_mutado = mutacion(hijo2, prob_mutacion_individuo, prob_mutacion_gen)
            poblacion.extend([hijo1_mutado, hijo2_mutado])
            print(f"Hijo 1 (antes de mutacion): {hijo1}, después de mutacion: {hijo1_mutado}")
            print(f"Hijo 2 (antes de mutacion): {hijo2}, después de mutacion: {hijo2_mutado}")
        
        
        print(f"Poblacion antes de la poda: {poblacion}")
        
        poblacion_enteros = [binario_a_entero(individuo) for individuo in poblacion]
        valores_x_poblacion = [calcular_valor_x(individuo, a, resolucion_nueva) for individuo in poblacion]
        valores_fx_poblacion = [calcular_fx(x) for x in valores_x_poblacion]
        
        mejor, peor, promedio = calcular_estadisticas(valores_fx_poblacion, maximizar)
        print(f"Antes de la poda - Generacion {generacion + 1}: Mejor: {mejor}, Peor: {peor}, Promedio: {promedio}")
        mejores.append(mejor)
        peores.append(peor)
        promedios.append(promedio)

        indice_mejor = valores_fx_poblacion.index(mejor)
        indice_peor = valores_fx_poblacion.index(peor)
        mejor_x = valores_x_poblacion[indice_mejor]
        peor_x = valores_x_poblacion[indice_peor]
        
        # Graficar y guardar la gráfica
        plt.figure(figsize=(10, 6))
        plt.plot(puntos_x, puntos_fx, label='Función Fitness', color='grey')
        
        # if generacion == 0:
        #     plt.scatter([], [], color='black', label='Resto de Individuos')
        
        for x, fx in zip(valores_x_poblacion, valores_fx_poblacion):
            plt.scatter(x, fx, color='black', zorder=3)
        plt.scatter([mejor_x], [mejor], color='green', zorder=5, label='Mejor Individuo')
        plt.scatter([peor_x], [peor], color='red', zorder=5, label='Peor Individuo')
        plt.title(f'Generación {generacion + 1}')
        plt.xlabel('Intervalo')
        plt.ylabel('F(x)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{carpeta_imagenes}/generacion_{generacion + 1}.png")
        plt.close()
        
        tamanos_poblacion.append(len(poblacion))
        poblacion = podar_poblacion(poblacion, poblacion_maxima, a, resolucion_nueva, maximizar)
        # tamanos_poblacion.append(len(poblacion))
        
        print(f"Poblacion después de la poda: {poblacion}")

    
        poblacion_enteros = [binario_a_entero(individuo) for individuo in poblacion]
        valores_x_poblacion = [calcular_valor_x(individuo, a, resolucion_nueva) for individuo in poblacion]
        valores_fx_poblacion = [calcular_fx(x) for x in valores_x_poblacion]
        
        
        mejorE, _, _ = calcular_estadisticas(poblacion_enteros, maximizar)
        mejorX, _, _ = calcular_estadisticas(valores_x_poblacion, maximizar)
        mejorFX, _, _ = calcular_estadisticas(valores_fx_poblacion, maximizar)
        # mejorI, _, _ = calcular_estadisticas(poblacion, maximizar)
        
        
        print(f"Poblacion en enteros: {poblacion_enteros}")
        print(f"Valores de X: {valores_x_poblacion}")
        print(f"Valores de f(x): {valores_fx_poblacion}")
        
        print('///////////////////////////////////////////////////////////// \n')
        print('ENTEROS. MEJOR: ', mejorE)
        print('X. MEJOR: ', mejorX)
        print('FX. MEJOR: ', mejorFX)
        # print('FX. MEJOR: ', mejorI)
        

    crear_video(carpeta_imagenes) #Video

    fig, ax = plt.subplots(figsize=(10, 6))
    generaciones = range(1, num_generaciones + 1)
    ax.plot(generaciones, mejores, label='Mejor', marker='o', linestyle='-', color='green')
    ax.plot(generaciones, peores, label='Peor', marker='o', linestyle='-', color='red')
    ax.plot(generaciones, promedios, label='Promedio', marker='o', linestyle='-', color='orange')
    ax.set_title('Evolución de la aptitud de los individuos')
    ax.set_xlabel('Generación')
    ax.set_ylabel('Fitness')
    ax.legend()
    ax.grid(True)
    plt.show()
    
    return poblacion, valores_x_poblacion, poblacion_enteros, valores_fx_poblacion

ventana = tk.Tk()
ventana.title("Algoritmo Genético")

tk.Label(ventana, text="Límite inferior del intervalo (a):").grid(row=0, column=0)
entrada_a = tk.Entry(ventana)
entrada_a.grid(row=0, column=1)

tk.Label(ventana, text="Límite superior del intervalo (b):").grid(row=1, column=0)
entrada_b = tk.Entry(ventana)
entrada_b.grid(row=1, column=1)

tk.Label(ventana, text="Resolucion:").grid(row=2, column=0)
resolucion_inicialE = tk.Entry(ventana)
resolucion_inicialE.grid(row=2, column=1)

tk.Label(ventana, text="Poblacion inicial:").grid(row=3, column=0)
poblacion_inicialE = tk.Entry(ventana)
poblacion_inicialE.grid(row=3, column=1)

tk.Label(ventana, text="Poblacion maxima:").grid(row=4, column=0)
poblacion_maximaE = tk.Entry(ventana)
poblacion_maximaE.grid(row=4, column=1)

tk.Label(ventana, text="No. de Generaciones/Iteraciones:").grid(row=5, column=0)
generaciones_ent = tk.Entry(ventana)
generaciones_ent.grid(row=5, column=1)

tk.Label(ventana, text="Probabilidad de cruza:").grid(row=6, column=0)
prob_cruzaE = tk.Entry(ventana)
prob_cruzaE.grid(row=6, column=1)

tk.Label(ventana, text="Probabilidad de mutacion (Individuo):").grid(row=7, column=0)
prob_ind = tk.Entry(ventana)
prob_ind.grid(row=7, column=1)

tk.Label(ventana, text="Probabilidad de mutacion (Gen):").grid(row=8, column=0)
prob_gen = tk.Entry(ventana)
prob_gen.grid(row=8, column=1)

opcion_maximizar = tk.IntVar(value=1)
tk.Radiobutton(ventana, text="Maximizar", variable=opcion_maximizar, value=1).grid(row=9, column=0)
tk.Radiobutton(ventana, text="Minimizar", variable=opcion_maximizar, value=2).grid(row=9, column=1)

boton_iniciar = tk.Button(ventana, text="Iniciar Algoritmo", command= algoritmo_genetico_con_seleccion_cruza)
boton_iniciar.grid(row=10, column=0, columnspan=2)

label_bits = tk.Label(ventana, text="Cantidad total de bits a usar: ")
label_bits.grid(row=11, column=0, columnspan=2)
label_resolucion = tk.Label(ventana, text="Nueva resolución: ")
label_resolucion.grid(row=12, column=0, columnspan=2)
label_saltos = tk.Label(ventana, text="Saltos/Puntos nuevos: ")
label_saltos.grid(row=13, column=0, columnspan=2)

ventana.mainloop()