ASF_WEIGHTS_FILE := SLOWFAST_EPIC.pyth
VSF_WEIGHTS_FILE := SLOWFAST.pyth
CONFIG_DIR := configs

EK_REPO_NAME := epic-kitchens-100-annotations
ES_REPO_NAME := epic-sounds-annotations
EK_DL_NAME := epic-kitchens-download-scripts
EK_NAME := epic-kitchens-100
DATA_DIR := data


.PHONY: data
data: # This target clones the repos only if they don't exist in the data directory
	@mkdir -p $(DATA_DIR) # Create the data directory if it doesn't exist

	@if [ ! -d "$(DATA_DIR)/$(EK_REPO_NAME)" ]; then \
		cd $(DATA_DIR) && git submodule add https://github.com/epic-kitchens/$(EK_REPO_NAME) ; \
	fi

	@if [ ! -d "$(DATA_DIR)/$(ES_REPO_NAME)" ]; then \
		cd $(DATA_DIR) && git submodule add https://github.com/epic-kitchens/$(ES_REPO_NAME) ; \
	fi

	@if [ ! -d "$(DATA_DIR)/$(EK_DL_NAME)" ]; then \
		cd $(DATA_DIR) && git submodule add https://github.com/epic-kitchens/$(EK_DL_NAME) ; \
	fi

	$(MAKE) update

.PHONY: update
update:
	@git submodule sync --recursive
	@git submodule update --init --recursive
	@git pull --recurse-submodules


.PHONY: weights
weights:
	@mkdir -p weights/asf weights/vsf
	@wget https://www.dropbox.com/s/cr0c6xdaggc2wzz/$(ASF_WEIGHTS_FILE) -O weights/asf/$(ASF_WEIGHTS_FILE)
	@wget https://www.dropbox.com/s/uxb6i2xkn91xqzi/$(VSF_WEIGHTS_FILE) -O weights/vsf/$(VSF_WEIGHTS_FILE)


.PHONY: bash
bash:
	@echo "Running interactive bash session"
	@srun --job-name "interactive bash" \
		--cpus-per-task 4 \
		--mem 32G \
		--time 4:00:00 \
		--pty bash


.PHONY: bash-gpu
bash-gpu:
	@echo "Running interactive bash session"
	@srun --job-name "bash-gpu" \
		--cpus-per-task 8 \
		--mem 32G \
		--gres gpu:1 \
		--time 4:00:00 \
		--pty bash


.PHONY: train
train:
	@echo "Running the main script"
	@./singrw <<< "python main.py --cfg ${CONFIG_DIR}/train-config.yaml --train"
