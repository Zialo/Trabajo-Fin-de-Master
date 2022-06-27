# make normal
normal:
	python3 dataset_generator_final.py 600 0.8 6001
	python3 dataset_generator_final.py 500 0.8 5001
	python3 dataset_generator_final.py 500 0.8 5002	     # [ITERATIONS], [TRAIN_SIZE]
	python3 dataset_generator_final.py 100 0.8 2	     # [ITERATIONS], [TRAIN_SIZE]
	python3 dataset_generator_final.py 100 0.8 3	     # [ITERATIONS], [TRAIN_SIZE]
	python3 dataset_generator_final.py 100 0.8 4	     # [ITERATIONS], [TRAIN_SIZE]
	python3 dataset_generator_final.py 100 0.8 5	     # [ITERATIONS], [TRAIN_SIZE]
	python3 dataset_generator_final.py 100 0.8 6	     # [ITERATIONS], [TRAIN_SIZE]
	python3 dataset_generator_final.py 100 0.8 7	     # [ITERATIONS], [TRAIN_SIZE]
	python3 dataset_generator_final.py 100 0.8 8	     # [ITERATIONS], [TRAIN_SIZE]
	python3 dataset_generator_final.py 100 0.8 9	     # [ITERATIONS], [TRAIN_SIZE]
	python3 dataset_generator_final.py 100 0.8 10	     # [ITERATIONS], [TRAIN_SIZE]


# make simulacion
simulacion:
	python3 Simulation_t-5.py 0 50 100 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
