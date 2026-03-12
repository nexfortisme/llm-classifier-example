package classifier

import (
	"fmt"
	"github.com/nexfortisme/classifier-example/embedder"
	"github.com/nexfortisme/classifier-example/store"
)

// Intent labels
const (
	IntentDirectRequest         = "direct_request"
	IntentBotDirectedExplicit   = "bot_directed_explicit"
	IntentBotDirectedByFollowup = "bot_directed_by_followup"
	IntentIndirectRequest       = "indirect_request"
	IntentMisdirected           = "misdirected"
	IntentConversationStarter   = "conversation_starter"
	IntentUnrelated             = "unrelated"
	IntentAbstract              = "abstract"
	IntentAmbiguous             = "ambiguous"
	IntentPassive               = "passive"
	IntentPositive              = "positive"
	IntentNegative              = "negative"
	IntentUtility               = "utility"
)

// Result is the output of a classification.
type Result struct {
	Intent     string
	Confidence float32
	TopMatches []store.QueryResult
}

// Classifier compares incoming messages against stored embeddings.
type Classifier struct {
	embedder  embedder.Embedder
	store     *store.MemoryStore
	topK      int
	threshold float32
}

func New(e embedder.Embedder, s *store.MemoryStore) *Classifier {
	return &Classifier{
		embedder:  e,
		store:     s,
		topK:      5,
		threshold: 0.70, // minimum similarity to count as a vote
	}
}

// WithTopK sets how many nearest neighbors to consider.
func (c *Classifier) WithTopK(k int) *Classifier {
	c.topK = k
	return c
}

// WithThreshold sets the minimum similarity score to count as a vote.
func (c *Classifier) WithThreshold(t float32) *Classifier {
	c.threshold = t
	return c
}

// Classify embeds the message and votes against stored examples.
func (c *Classifier) Classify(message string) (Result, error) {
	if c.store.Len() == 0 {
		return Result{}, fmt.Errorf("store is empty — run the seed command first")
	}

	vec, err := c.embedder.Embed(message)
	if err != nil {
		return Result{}, fmt.Errorf("embed failed: %w", err)
	}

	matches := c.store.Query(vec, c.topK)

	// Weighted vote — each neighbor contributes its similarity score
	votes := map[string]float32{}
	for _, match := range matches {
		if match.Similarity >= c.threshold {
			votes[match.Entry.Intent] += match.Similarity
		}
	}

	// Find the winning intent
	var winner string
	var topScore float32
	for intent, score := range votes {
		if score > topScore {
			topScore = score
			winner = intent
		}
	}

	// Nothing cleared the threshold
	if winner == "" {
		winner = IntentAmbiguous
	}

	return Result{
		Intent:     winner,
		Confidence: topScore,
		TopMatches: matches,
	}, nil
}
