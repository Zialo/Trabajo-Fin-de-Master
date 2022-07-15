# make normal
normal:
	python3 dataset_generator_final.py 500 100 0.8 data_4	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	
# make different_task
stick_puss:
	python3 dataset_generator_stick_puss.py 10 150 0.5 data	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]

button_press:
	python3 dataset_generator_button_press.py 200 100 0.8 button1	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]
	python3 dataset_generator_button_press.py 200 100 0.8 button2	     # [ITERATIONS], [FRAMES_PER_IT], [TRAIN_SIZE]


# make simulacion
simulacion_t-5:
	python3 Simulation_t-5.py 0 50 100 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_t-5_button:
	python3 Simulation_basic_v2_button.py 0 50 100 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]

simulacion_basic:
	python3 Simulation_basic.py 0 100 150 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_basic_v2:
	python3 Simulation_basic_v2.py 0 100 150 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_basic_v2_ng:
	python3 Simulation_basic_v2_noGripper.py 0 100 150 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
	
simulacion_t-5_button_ng:
	python3 Simulation_basic_v2_button_noGripper.py 0 100 100 1 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [SAVE_IMAGES], [COPPELIA], [MODELO]
