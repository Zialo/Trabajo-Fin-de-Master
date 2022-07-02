# make normal
normal:
	python3 dataset_generator_final.py 500 100 0.8 data_4	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]

# make simulacion
simulacion_t-5:
	python3 Simulation_t-5.py 0 50 100 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]

simulacion_basic:
	python3 Simulation_basic.py 0 3 60 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
