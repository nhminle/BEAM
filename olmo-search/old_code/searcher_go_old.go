package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Record holds the JSONL record from a shard.
type Record struct {
	Text string `json:"text"`
}

// resultWriter serializes writes to result files.
type resultWriter struct {
	mu sync.Mutex
}

// writeAggregatedResults writes out the final results for one CSV file.
// It overwrites any previous content.
func (w *resultWriter) writeAggregatedResults(csvPath, outputDir string, results []csvResult) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	outputPath := filepath.Join(outputDir,
		fmt.Sprintf("%s_results.csv",
			strings.TrimSuffix(filepath.Base(csvPath), ".csv"),
		))
	// Open file for writing (create or truncate)
	file, err := os.OpenFile(outputPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	if err := writer.Write([]string{"text", "shard_path"}); err != nil {
		return err
	}

	// Write all aggregated rows.
	for _, res := range results {
		if err := writer.Write([]string{res.Text, res.ShardPath}); err != nil {
			return err
		}
	}
	return nil
}

// -------------------------
// Aggregator types and globals
// -------------------------

// csvResult represents one matching row from a shard.
type csvResult struct {
	Text      string
	ShardPath string
}

var (
	aggregatorMu      sync.Mutex
	csvAggregator     = make(map[string][]csvResult) // key: CSV file path
	csvProcessedCount = make(map[string]int)         // key: CSV file path, number of shards that have processed it
)

// aggregatorUpdate appends new results for a given CSV file and increments its count.
// When a CSV file has been processed by all shards, its aggregated results are saved.
func aggregatorUpdate(csvPath string, newResults []csvResult, totalShardCount int, writer *resultWriter, outputDir string) {
	var resultsToWrite []csvResult

	aggregatorMu.Lock()
	// Append any new results.
	csvAggregator[csvPath] = append(csvAggregator[csvPath], newResults...)
	// Increment the count (even if there were no new results).
	csvProcessedCount[csvPath]++
	// If this CSV file has been processed by all shards...
	if csvProcessedCount[csvPath] == totalShardCount {
		// Copy results (so we can unlock before writing to disk).
		resultsToWrite = append([]csvResult(nil), csvAggregator[csvPath]...)
		// Clean up the maps.
		delete(csvAggregator, csvPath)
		delete(csvProcessedCount, csvPath)
	}
	aggregatorMu.Unlock()

	// If ready, write the aggregated results.
	if resultsToWrite != nil {
		if err := writer.writeAggregatedResults(csvPath, outputDir, resultsToWrite); err != nil {
			fmt.Printf("Error writing aggregated results for %s: %v\n", csvPath, err)
		} else {
			fmt.Printf("Saved aggregated results for %s\n", csvPath)
		}
	}
}

// -------------------------
// File discovery and processing functions
// -------------------------

// findShardFiles finds all files with a .jsonl extension under root.
func findShardFiles(root string) ([]string, error) {
	var shards []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".jsonl") {
			shards = append(shards, path)
		}
		return nil
	})
	return shards, err
}

// findCSVFiles finds all files with a .csv extension under root.
func findCSVFiles(root string) ([]string, error) {
	var csvFiles []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".csv") {
			csvFiles = append(csvFiles, path)
		}
		return nil
	})
	return csvFiles, err
}

// buildTextSet builds a set of texts from a shard (JSONL) file.
func buildTextSet(shardPath string) (map[string]struct{}, error) {
	file, err := os.Open(shardPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	textSet := make(map[string]struct{})
	decoder := json.NewDecoder(file)
	for decoder.More() {
		var r Record
		if err := decoder.Decode(&r); err != nil {
			// Skip decode errors.
			continue
		}
		if r.Text != "" {
			textSet[r.Text] = struct{}{}
		}
	}
	return textSet, nil
}

// searchCSV reads csvPath, looks in the given columns for texts that exist in textSet,
// and returns any matching texts.
func searchCSV(csvPath string, columns []string, textSet map[string]struct{}) ([]string, error) {
	file, err := os.Open(csvPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	headers, err := reader.Read()
	if err != nil {
		return nil, err
	}

	// Identify which column indices to search (case-insensitive).
	var colIndices []int
	for _, col := range columns {
		for i, header := range headers {
			if strings.EqualFold(header, col) {
				colIndices = append(colIndices, i)
				break
			}
		}
	}

	var matches []string
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		for _, idx := range colIndices {
			if idx < len(record) {
				text := strings.TrimSpace(record[idx])
				if _, exists := textSet[text]; exists {
					matches = append(matches, text)
				}
			}
		}
	}
	return matches, nil
}

// processShard processes one shard (JSONL file):
// 1. It builds the set of texts from the shard.
// 2. For each CSV file (book), it searches for matches.
// 3. Instead of writing immediately, it sends results to the aggregator.
// The totalShardCount is passed so the aggregator knows when a CSV file is complete.
func processShard(shardPath string, csvFiles []string, textCols []string, outputDir string, writer *resultWriter, totalShardCount int) {
	fmt.Printf("Processing shard: %s\n", shardPath)
	textSet, err := buildTextSet(shardPath)
	if err != nil {
		fmt.Printf("Error building text set for %s: %v\n", shardPath, err)
		return
	}

	var wg sync.WaitGroup
	// Limit concurrent CSV processing within this shard.
	workerSem := make(chan struct{}, 10)

	for _, csvFile := range csvFiles {
		wg.Add(1)
		workerSem <- struct{}{}
		go func(csvPath string) {
			defer func() {
				<-workerSem
				wg.Done()
			}()

			fmt.Printf("Processing CSV file: %s (in shard: %s)\n", csvPath, shardPath)
			matches, err := searchCSV(csvPath, textCols, textSet)
			if err != nil {
				fmt.Printf("Error searching %s: %v\n", csvPath, err)
				// Even on error, record that this shard processed the CSV.
				aggregatorUpdate(csvPath, nil, totalShardCount, writer, outputDir)
				return
			}

			// Convert matches to csvResult entries.
			var results []csvResult
			for _, m := range matches {
				results = append(results, csvResult{Text: m, ShardPath: shardPath})
			}
			// Update the aggregator for this CSV file.
			aggregatorUpdate(csvPath, results, totalShardCount, writer, outputDir)
		}(csvFile)
	}
	wg.Wait()
}

func main() {
	// Adjust these paths as needed.
	baseDir := "/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/"
	shardRoot := "/scratch3/workspace/ekorukluoglu_umass_edu-simple/shard/"
	outputDir := "./results"
	textColumns := []string{"en", "tr", "es", "vi"}

	// Create the output directory.
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Printf("Error creating output dir: %v\n", err)
		return
	}

	// Find all CSV (book) files recursively.
	csvFiles, err := findCSVFiles(baseDir)
	if err != nil {
		fmt.Printf("Error finding CSV files: %v\n", err)
		return
	}
	fmt.Printf("Found %d CSV files\n", len(csvFiles))

	// Find all shard files.
	shardFiles, err := findShardFiles(shardRoot)
	if err != nil {
		fmt.Printf("Error finding shard files: %v\n", err)
		return
	}
	totalShardCount := len(shardFiles)
	fmt.Printf("Found %d shards\n", totalShardCount)

	// Prepare a result writer.
	var writer resultWriter

	// Process shards concurrently.
	var shardWg sync.WaitGroup
	shardCh := make(chan string, totalShardCount)

	// Start a fixed number of shard workers (e.g. 25).
	numShardWorkers := 40
	for i := 0; i < numShardWorkers; i++ {
		go func() {
			for shardPath := range shardCh {
				processShard(shardPath, csvFiles, textColumns, outputDir, &writer, totalShardCount)
				shardWg.Done()
			}
		}()
	}

	// Enqueue all shard files.
	for _, shard := range shardFiles {
		shardWg.Add(1)
		shardCh <- shard
	}
	close(shardCh)

	// Wait for all shards to finish processing.
	shardWg.Wait()
	fmt.Printf("Processed all %d shards. Aggregated results have been saved per CSV (book) in: %s\n", totalShardCount, outputDir)
}
