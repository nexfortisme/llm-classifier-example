package store

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
)

// Entry is a single stored embedding with its label and source text.
type Entry struct {
	ID     string
	Text   string
	Intent string
	Vector []float32
}

// QueryResult is a nearest-neighbor match returned from a query.
type QueryResult struct {
	Entry      Entry
	Similarity float32
}

// MemoryStore holds all embeddings in memory with optional persistence to disk.
type MemoryStore struct {
	mu      sync.RWMutex
	entries []Entry
}

func New() *MemoryStore {
	return &MemoryStore{}
}

// Add stores a new embedding entry.
func (s *MemoryStore) Add(id, text, intent string, vector []float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = append(s.entries, Entry{
		ID:     id,
		Text:   text,
		Intent: intent,
		Vector: vector,
	})
}

// Query returns the top-k most similar entries to the given vector.
func (s *MemoryStore) Query(vector []float32, topK int) []QueryResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	results := make([]QueryResult, 0, len(s.entries))
	for _, entry := range s.entries {
		sim := cosineSimilarity(vector, entry.Vector)
		results = append(results, QueryResult{Entry: entry, Similarity: sim})
	}

	// Sort descending by similarity
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if topK > len(results) {
		topK = len(results)
	}
	return results[:topK]
}

// Len returns the number of stored entries.
func (s *MemoryStore) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

// -- Persistence --

type persistedEntry struct {
	ID     string    `json:"id"`
	Text   string    `json:"text"`
	Intent string    `json:"intent"`
	Vector []float32 `json:"vector"`
}

// Save writes all entries to a JSON file on disk.
func (s *MemoryStore) Save(path string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	persisted := make([]persistedEntry, len(s.entries))
	for i, e := range s.entries {
		persisted[i] = persistedEntry{
			ID:     e.ID,
			Text:   e.Text,
			Intent: e.Intent,
			Vector: e.Vector,
		}
	}

	data, err := json.MarshalIndent(persisted, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal store: %w", err)
	}

	return os.WriteFile(path, data, 0644)
}

// Load reads entries from a previously saved JSON file.
func (s *MemoryStore) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read store file: %w", err)
	}

	var persisted []persistedEntry
	if err := json.Unmarshal(data, &persisted); err != nil {
		return fmt.Errorf("unmarshal store: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.entries = make([]Entry, len(persisted))
	for i, p := range persisted {
		s.entries[i] = Entry{
			ID:     p.ID,
			Text:   p.Text,
			Intent: p.Intent,
			Vector: p.Vector,
		}
	}

	return nil
}

// -- Math --

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}
