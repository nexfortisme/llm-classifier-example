package main

import (
	"github.com/nexfortisme/classifier-example/embedder"
	"github.com/nexfortisme/classifier-example/store"
	"encoding/json"
	"fmt"
	"os"
)

type Example struct {
	Text   string `json:"text"`
	Intent string `json:"intent"`
}

func main() {
	examplesPath := "examples.json"
	storePath := "store.json"

	if len(os.Args) > 1 {
		examplesPath = os.Args[1]
	}
	if len(os.Args) > 2 {
		storePath = os.Args[2]
	}

	// -- Load examples --
	data, err := os.ReadFile(examplesPath)
	if err != nil {
		fmt.Printf("error: could not read %s: %v\n", examplesPath, err)
		os.Exit(1)
	}

	var examples []Example
	if err := json.Unmarshal(data, &examples); err != nil {
		fmt.Printf("error: could not parse %s: %v\n", examplesPath, err)
		os.Exit(1)
	}

	fmt.Printf("loaded %d examples from %s\n\n", len(examples), examplesPath)

	// -- Set up embedder --
	emb := embedder.NewLMStudio(
		"http://127.0.0.1:1234",
		"text-embedding-nomic-embed-text-v1.5",
	)

	// -- Set up store --
	st := store.New()

	// -- Embed each example and store it --
	for i, ex := range examples {
		fmt.Printf("[%d/%d] embedding [%s]: %q\n", i+1, len(examples), ex.Intent, ex.Text)

		vec, err := emb.Embed(ex.Text)
		if err != nil {
			fmt.Printf("  warning: skipping — %v\n", err)
			continue
		}

		st.Add(fmt.Sprintf("%d", i), ex.Text, ex.Intent, vec)
	}

	fmt.Printf("\nstored %d embeddings\n", st.Len())

	// -- Persist to disk --
	if err := st.Save(storePath); err != nil {
		fmt.Printf("error: could not save store: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("saved store to %s\n", storePath)
	fmt.Println("seeding complete — run the classify command to test")
}
