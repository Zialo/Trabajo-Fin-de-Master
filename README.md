<img src='http://canal.ugr.es/wp-content/uploads/2017/07/logo-UGR-color-vertical.jpg' width=15% align="right" />

# Trabajo de Fin de Master
Repositorio dedicado a almacenar los recursos necesarios para la realización del Trabajo de Fin de Master

## Nombre del Proyecto
Análisis y estudio en frameworks de simulación sobre la generación de datos en problemas robótica industrial y su aplicación en tareas de aprendizaje automático. 

## Descripción del problema:
La robótica industrial es una rama de la ingeniería que pretende realizar múltiples procesos industriales tales como la manipulación de objetos haciendo uso de robots con el objetivo de completar diversas tareas en cadena de forma automática de modo que necesitemos la mínima supervisión humana posible. Partiendo de esta definición, podemos encontrarnos tanto con tareas que no dependan del entorno y que por tanto, se realicen aplicando los mismos movimientos sobre el robot como con otras en las que tanto el entorno como los objetos u obstáculos que se encuentren en él sean cambiantes, por lo que necesitemos hacer que nuestro robot adapte sus movimientos, decisiones o incluso fuerza en base a lo que se encuentre en cada momento. Para ello, debemos hacer que el robot consiga manejarse en distintas situaciones, haciendo que aprenda de una cantidad de escenarios lo suficientemente amplia como para que sepa cómo actuar frente a un obstáculo inesperado nunca antes visto o aprendido en la fase de entrenamiento. Es aquí donde cobra una importancia real la realización de un estudio minucioso sobre la generación de datos como pueden ser distintos escenarios, la clase de imágenes sobre el entorno que van a ser capaces de aportar una mayor información, el tipo de acciones con las que el robot va a ser capaz de generalizar mejor el problema…
En este contexto, es común simular distintas tareas que pueda realizar un brazo robótico tales como pueden ser coger una pelota, apilar unos cubos, entre otros, y evaluar los datos devueltos en busca de los casos de estudio más adecuados. A partir de los mismos, se realizará un proceso de entrenamiento para generar un modelo que sea capaz de replicar estas tareas en sus simuladores correspondientes. Adicionalmente, se estudiará la mejor sinergia entre caso de estudio y simulador con vistas a formular una generalización de los tipos de datos que se han de utilizar para futuros problemas relacionados con la robótica en el ámbito de la inteligencia artificial.

## Objetivos del problema

El objetivo principal será obtener una configuración general de datos con el que se caractericen distintos casos de uso en robótica industrial. Para ello, desglosamos entre los siguientes subobjetivos:

1. Estudiar y analizar en profundidad tanto los distintos estudios y proyectos realizados en este ámbito como los diferentes frameworks de simulación utilizados, dejando claras tanto sus diferencias como sus similitudes. 

2. Generar datos para un conjunto de tareas con cada uno de los frameworks que sean los que utilicemos para generar los distintos modelos en la fase de entrenamiento.

3. Analizar las mejores configuraciones posibles de datos en busca de un patrón capaz de generar una generalización para futuros problemas de modo que sepamos de donde partir inicialmente sea cual sea el problema a evaluar. Para esto trataremos de encontrar las configuraciones que maximicen el RMSE como métrica estándar.

4. Simular en los robots los modelos generados y visualiza los resultados obtenidos mediante diferentes aproximaciones. 

## Frameworks 

### CoppeliaSim
  1. RLBench: https://github.com/stepjam/RLBench



### MuJoCo
  1. MetaWorld: https://github.com/rlworkgroup/metaworld
  2. SURREAL: https://github.com/SurrealAI/surreal
  3. OpenAI Gym: https://github.com/openai/gym
