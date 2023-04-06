import cv2
import numpy
import mss
import math
import networkx as nx
from queue import PriorityQueue
from collections import deque
import pyautogui
import time


template_apple = cv2.imread('manzana2.jpg', cv2.IMREAD_GRAYSCALE)
w, h = template_apple.shape[::-1]
direction=0
#matrix=numpy.empty((15,17),dtype=str)
snake = deque()
snake.append((7,4))
snake.append((7,3))
snake.append((7,2))
snake.append((7,1))
matrix=numpy.zeros((15,17),dtype=int)
apple_pos=(7,12)
head_init=(7,4)
aux_apple_pos =(7,12)
#matrix[apple_pos[0]][apple_pos[1]]="M"
matrix[apple_pos[0]][apple_pos[1]]=1
#matrix[7][4]="C"
matrix[7][12]=2
#matrix[7][1:4]="S"
matrix[7][9:12]=3
size=4
direction = 1


# Creamos un grafo vacío
grafo = nx.Graph()
def dijkstra(start, end, graph):
    # Creamos un diccionario para almacenar las distancias desde el nodo de origen hasta cada nodo del grafo
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Creamos un diccionario para almacenar el camino más corto hasta cada nodo del grafo
    previous_nodes = {node: None for node in graph}

    # Creamos una cola de prioridad para almacenar los nodos que se deben visitar, con la distancia más corta en primer lugar
    queue = PriorityQueue()
    queue.put((0, start))

    while not queue.empty():
        # Obtenemos el nodo actual y su distancia desde el nodo de origen
        current_distance, current_node = queue.get()

        # Si llegamos al nodo destino, devolvemos el camino más corto
        if current_node == end:
            path = []
            while previous_nodes[current_node] is not None:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            path.append(start)
            path.reverse()
            return path

        # Recorremos los nodos adyacentes al nodo actual
        for neighbor in graph[current_node]:
            # Calculamos la nueva distancia para llegar al nodo adyacente
            distance = current_distance + 1

            # Si la nueva distancia es menor que la distancia almacenada, actualizamos la distancia y el nodo anterior
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                queue.put((distance, neighbor))

    # Si no se puede llegar al nodo destino desde el nodo de origen, devolvemos None
    return None

def create_graph(matrix_game):
    graph = {}
    for i in range(matrix_game.shape[0]):
        for j in range(matrix_game.shape[1]):
            node = (i, j)
            neighbors = []
            if i > 0:
                if matrix[i-1][j] == 0 or matrix[i-1][j] == 1 :
                    neighbors.append((i-1, j))
            if i < matrix_game.shape[0] - 1:
                if matrix[i+1][j] == 0 or matrix[i+1][j] == 1:
                    neighbors.append((i+1, j))
                
            if j > 0:
                if matrix[i][j-1] == 0 or matrix[i][j-1] == 1:
                    neighbors.append((i, j-1))
                
            if j < matrix_game.shape[1] - 1:
                if matrix[i][j+1] == 0 or matrix[i][j+1] == 1:
                    neighbors.append((i, j+1))
            
            graph[node] = neighbors
    return graph

def compute(cor_x_apple, cor_y_apple):
    global direccion
    global apple_pos
    global head_init
    global grafo
    global matrix
    global aux_apple_pos
    global size


    apple_pos=(math.floor(cor_y_apple/32)-1,math.floor(cor_x_apple/32)-1)

    #matrix[apple_pos[0]][apple_pos[1]] = "M"
    matrix[apple_pos[0]][apple_pos[1]] = 1

    print(apple_pos[0],apple_pos[1])
    graph = create_graph(matrix)

    

    if aux_apple_pos != apple_pos:
        head_init = aux_apple_pos
        aux_apple_pos = apple_pos
        size +=1
        path = dijkstra(head_init, apple_pos, graph)
        
        move(path)
        snake_size = len(snake)
        min_snake = min(size, len(path)-1)
        for i in range(len(path)-1):
            if i >= snake_size :
                break
            snake.pop()
        list_snake = deque(snake)
        snake.clear()
        for i in range(1,min_snake+1):
            snake.append(path[-i])
        snake.extend(list_snake)
        matrix = numpy.where(matrix == 3, 0, matrix)
        for i in range(len(snake)):
            matrix[snake[i][0]][snake[i][1]] = 3
        print(path)
        #print("____________________")
        #print(snake)
        #print(list_snake)

def move(path):
    time2 = 0 
    if len(path) > 10:
        time2 = 0.04

    for i in range(len(path)-1):
        if path[i][0] > path[i+1][0]:
            print("up")
            pyautogui.press("up")
            time.sleep(time2)
        elif path[i][0] < path[i+1][0]:
            print("down")
            pyautogui.press("down")
            time.sleep(time2)
        elif path[i][1] > path[i+1][1]:
            print("left")
            time.sleep(time2)
            pyautogui.press("left")
        elif path[i][1] < path[i+1][1]:
            print("right")
            pyautogui.press("right")
            time.sleep(time2)




"""
    # Añadimos los nodos al grafo
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            grafo.add_node(matrix[i][j])

    # Recorremos la matriz de adyacencia para añadir las aristas al grafo
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i < matrix.shape[0] - 1:
                # Añadimos la arista con el nodo de la casilla de abajo
                grafo.add_edge(matrix[i][j], matrix[i + 1][j])
            if j < matrix.shape[1] - 1:
                # Añadimos la arista con el nodo de la casilla de la derecha
                grafo.add_edge(matrix[i][j], matrix[i][j + 1])
"""


with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 180, "left": 0, "width": 600, "height": 520}

    while "Screen capturing":

        img = numpy.array(sct.grab(monitor))
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(gris, (0, 100, 20), (8, 255, 255))
        res_apple = cv2.matchTemplate(mask1, template_apple, cv2.TM_CCORR_NORMED)
        min_val_apple, max_val_apple, min_loc_apple, max_loc_apple = cv2.minMaxLoc(res_apple)
        threshold = 0.84
        if (max_val_apple>threshold):
            top_left_apple = max_loc_apple
            bottom_right_apple = (top_left_apple[0] + w, top_left_apple[1] + h)
            cor_x_apple = (top_left_apple[0] + bottom_right_apple[0]) / 2
            cor_y_apple = (top_left_apple[1] + bottom_right_apple[1]) / 2
            #cv2.rectangle(img, (30, 30), (570, 510), 255, 2)
            #cv2.circle(img, (int(cor_x_Manzana), int(cor_y_Manzana)), 2, (0, 0, 255), -1)
            
            compute(cor_x_apple, cor_y_apple)
        #cv2.imshow('juego', img)
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break