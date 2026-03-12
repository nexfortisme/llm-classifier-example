BINARY_DIR=bin

.PHONY: all build seed classify clean

all: build

build:
	@mkdir -p $(BINARY_DIR)
	go build -o $(BINARY_DIR)/seed ./cmd/seed
	go build -o $(BINARY_DIR)/classify ./cmd/classify
	@echo "built bin/seed and bin/classify"

# Phase 1: embed examples.json and save store.json
seed:
	@$(BINARY_DIR)/seed examples.json store.json

# Phase 2: interactive classifier loop
classify:
	@$(BINARY_DIR)/classify store.json

clean:
	rm -rf $(BINARY_DIR) store.json
