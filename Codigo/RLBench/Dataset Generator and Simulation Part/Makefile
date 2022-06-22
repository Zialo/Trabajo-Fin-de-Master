# make normal
normal:
	python3 dataset_generator.py 2000 1 reach_target	     # [ITERATIONS], [VARIATIONS], [TASK]
	python3 Data_getter.py 1 1 reach_target        	     # [RESERVED], [TASK], [NORMAL]
	python3 Train_test_split_multimage.py 0.80 1 reach_target  # [% TRAIN], [RESERVED], [TASK]

# make subpruebas	
subpruebas:
	python3 dataset_generator.py 1000 1 reach_target           # [ITERATIONS], [VARIATIONS], [TASK]
	python3 Data_getter.py 1 0 reach_target		     # [SUBPRUEBAS], [NOMBRE_CARPETA]
	python3 Train_only_multimage.py 72406 2000_Pruebas reach_target

# make gif MODEL=[MODEL_NAME]
gif:
	python3 Model_loader_six_channels_three_prevs.py 1 20 200 0 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [COPPELIA], [SAVE_IMAGES], [MODELO]
	python3 Gif_maker_final.py 
	
analisis:
	python3 Model_loader_six_channels_three_prevs.py 1 100 200 0 True $(MODEL)  # [NORMALIZED_DATA], [NUM_PRUEBAS], [ITERATIONS], [COPPELIA], [SAVE_IMAGES], [MODELO]
