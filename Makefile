ASF_WEIGHTS_FILE := SLOWFAST_EPIC.pyth
VSF_WEIGHTS_FILE := SlowFast.pyth

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

.PHONY: clone
clone:
	@git submodule add git@github.com:ClementSicard/auditory-slow-fast.git asf
	@git submodule add git@github.com:ClementSicard/slowfast.git vsf


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
