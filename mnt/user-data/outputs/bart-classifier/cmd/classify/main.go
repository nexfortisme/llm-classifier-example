package main

import (
	"github.com/nexfortisme/classifier-example/classifier"
	"github.com/nexfortisme/classifier-example/embedder"
	"github.com/nexfortisme/classifier-example/store"
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	storePath := "store.json"
	if len(os.Args) > 1 && !strings.HasPrefix(os.Args[1], "-") {
		storePath = os.Args[1]
	}

	// -- Set up embedder --
	emb := embedder.NewLMStudio(
		"http://127.0.0.1:1234",
		"text-embedding-nomic-embed-text-v1.5",
	)

	// -- Load store from disk --
	st := store.New()
	if err := st.Load(storePath); err != nil {
		fmt.Printf("error: could not load store from %s: %v\n", storePath, err)
		fmt.Println("hint: run the seed command first")
		os.Exit(1)
	}

	fmt.Printf("loaded %d embeddings from %s\n\n", st.Len(), storePath)

	// -- Set up classifier --
	clf := classifier.New(emb, st).
		WithTopK(5).
		WithThreshold(0.70)

	// -- Interactive loop --
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("enter a message to classify (ctrl+c to quit):")
	fmt.Println(strings.Repeat("-", 50))

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		message := strings.TrimSpace(scanner.Text())
		if message == "" {
			continue
		}

		result, err := clf.Classify(message)
		if err != nil {
			fmt.Printf("error: %v\n\n", err)
			continue
		}

		printResult(message, result)
	}
}

func printResult(message string, result classifier.Result) {
	fmt.Printf("\nmessage:    %q\n", message)
	fmt.Printf("intent:     %s\n", formatIntent(result.Intent))
	fmt.Printf("confidence: %.4f\n", result.Confidence)
	fmt.Println("\ntop matches:")

	for i, match := range result.TopMatches {
		fmt.Printf("  %d. [%.4f] [%s] %q\n",
			i+1,
			match.Similarity,
			match.Entry.Intent,
			match.Entry.Text,
		)
	}

	fmt.Println()

	// Application routing decision
	switch result.Intent {
	case classifier.IntentDirect:
		fmt.Println("→ ROUTE: send to LLM for response")
	case classifier.IntentNegative:
		fmt.Println("→ ROUTE: discard / do not prompt LLM")
	case classifier.IntentMeta:
		fmt.Println("→ ROUTE: log as observation / pass through")
	case classifier.IntentAmbiguous:
		fmt.Println("→ ROUTE: request clarification from user")
	default:
		fmt.Println("→ ROUTE: unknown — review manually")
	}

	fmt.Println(strings.Repeat("-", 50))
}

func formatIntent(intent string) string {
	labels := map[string]string{
		classifier.IntentDirect:    "✅ direct",
		classifier.IntentNegative:  "❌ negative",
		classifier.IntentMeta:      "💬 meta",
		classifier.IntentAmbiguous: "❓ ambiguous",
	}
	if label, ok := labels[intent]; ok {
		return label
	}
	return intent
}
