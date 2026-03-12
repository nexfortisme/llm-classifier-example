package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/nexfortisme/classifier-example/embedder"
	"github.com/nexfortisme/classifier-example/store"
)

type Example struct {
	Text   string `json:"text"`
	Intent string `json:"intent"`
}

type bartMessage struct {
	Turn int    `json:"turn"`
	Text string `json:"text"`
}

type bartExample struct {
	Type     string        `json:"type"`
	Messages []bartMessage `json:"messages"`
}

type bartDataset struct {
	Examples []bartExample `json:"examples"`
}

func loadExamples(path string) ([]Example, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("could not read %s: %w", path, err)
	}

	// Detect format: bart dataset is a JSON object with an "examples" key
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err == nil {
		if _, ok := raw["examples"]; ok {
			var ds bartDataset
			if err := json.Unmarshal(data, &ds); err != nil {
				return nil, fmt.Errorf("could not parse bart dataset %s: %w", path, err)
			}
			return convertBartDataset(ds), nil
		}
	}

	// Fall back to flat array format
	var examples []Example
	if err := json.Unmarshal(data, &examples); err != nil {
		return nil, fmt.Errorf("could not parse %s: %w", path, err)
	}
	return examples, nil
}

func convertBartDataset(ds bartDataset) []Example {
	examples := make([]Example, 0, len(ds.Examples))
	for _, ex := range ds.Examples {
		var parts []string
		for _, msg := range ex.Messages {
			parts = append(parts, msg.Text)
		}
		examples = append(examples, Example{
			Text:   strings.Join(parts, "\n"),
			Intent: ex.Type,
		})
	}
	return examples
}

func main() {
	// examplesPath := "bart_intent_dataset_expanded.json"
	// examplesPath := "test_embeddings_chatbot_interaction_dataset.json"
	// examplesPath := "test_embeddings_chatbot_interaction_dataset_v2.json"
	// examplesPath := "test_embeddings_chatbot_interaction_dataset_v3_names_expanded.json"
	examplesPath := "test_embeddings_chatbot_interaction_dataset_v4_discord_noisy.json"
	storePath := "store.json"

	if len(os.Args) > 1 {
		examplesPath = os.Args[1]
	}
	if len(os.Args) > 2 {
		storePath = os.Args[2]
	}

	examples, err := loadExamples(examplesPath)
	if err != nil {
		fmt.Printf("error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("loaded %d examples from %s\n\n", len(examples), examplesPath)

	// -- Set up embedder --
	emb := embedder.NewLMStudio(
		// "http://127.0.0.1:1234",
		"http://100.117.31.96:1234",
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
