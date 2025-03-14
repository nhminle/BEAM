package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
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

// appendShardResults opens (or creates) a text file for the given CSV file (book) and appends the results.
// If the file does not exist (or is empty), it writes a header line.
func (w *resultWriter) appendShardResults(csvPath, outputDir string, results []csvResult) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	outputPath := filepath.Join(outputDir,
		fmt.Sprintf("%s_results.txt",
			strings.TrimSuffix(filepath.Base(csvPath), ".csv"),
		))
	// Check if the file exists.
	_, err := os.Stat(outputPath)
	fileExists := err == nil

	// Open the file in append mode (or create it).
	file, err := os.OpenFile(outputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	// If the file is new, write the header.
	if !fileExists {
		if _, err := file.WriteString("text,shard_path\n"); err != nil {
			return err
		}
	}

	// Append each result.
	for _, res := range results {
		line := fmt.Sprintf("%s,%s\n", res.Text, res.ShardPath)
		if _, err := file.WriteString(line); err != nil {
			return err
		}
	}
	return nil
}

// -------------------------
// Aggregator types and globals (removed aggregator aggregation)
// -------------------------

// csvResult represents one matching row from a shard.
type csvResult struct {
	Text      string
	ShardPath string
}

// -------------------------
// Fuzzy matching functions
// -------------------------

// levenshtein calculates the Levenshtein distance between two strings.
func levenshtein(a, b string) int {
	la, lb := len(a), len(b)
	// Create a 2D slice.
	dp := make([][]int, la+1)
	for i := 0; i <= la; i++ {
		dp[i] = make([]int, lb+1)
		dp[i][0] = i
	}
	for j := 0; j <= lb; j++ {
		dp[0][j] = j
	}

	for i := 1; i <= la; i++ {
		for j := 1; j <= lb; j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}
			dp[i][j] = int(math.Min(
				math.Min(float64(dp[i-1][j])+1, float64(dp[i][j-1])+1),
				float64(dp[i-1][j-1]+cost),
			))
		}
	}
	return dp[la][lb]
}

// fuzzyMatch returns true if the similarity ratio between s1 and s2 is at least 80%.
func fuzzyMatch(s1, s2 string) bool {
	distance := levenshtein(s1, s2)
	maxLen := math.Max(float64(len(s1)), float64(len(s2)))
	if maxLen == 0 {
		return true
	}
	ratio := 1 - float64(distance)/maxLen
	return ratio >= 0.8
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

// searchCSV reads csvPath, looks in the given columns for texts that fuzzy match
// any key in textSet (80% similarity), and returns any matching texts.
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
				// Fuzzy match: iterate over keys in textSet.
				for key := range textSet {
					if fuzzyMatch(key, text) {
						matches = append(matches, text)
						break // Found a match; move to the next column.
					}
				}
			}
		}
	}
	return matches, nil
}

// processShard processes one shard (JSONL file):
// 1. It builds the set of texts from the shard.
// 2. For each CSV file (book), it searches for matches.
// 3. Instead of waiting for all shards to finish, it appends the results to a text file immediately.
func processShard(shardPath string, csvFiles []string, textCols []string, outputDir string, writer *resultWriter) {
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
				return
			}

			// Convert matches to csvResult entries.
			var results []csvResult
			for _, m := range matches {
				results = append(results, csvResult{Text: m, ShardPath: shardPath})
			}

			// Immediately append the results for this CSV file.
			if len(results) > 0 {
				if err := writer.appendShardResults(csvPath, outputDir, results); err != nil {
					fmt.Printf("Error appending results for %s: %v\n", csvPath, err)
				} else {
					fmt.Printf("Appended results for %s from shard %s\n", csvPath, shardPath)
				}
			}
		}(csvFile)
	}
	wg.Wait()
}

func main() {
	// Adjust these paths as needed.
	baseDir := "/home/ekorukluoglu_umass_edu/beam2/BEAM/olmo-search/sample-set/"
	shardRoot := "/scratch3/workspace/ekorukluoglu_umass_edu-simple/shard/global-shard_01_of_10/local-shard_0_of_10/"
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
	fmt.Printf("Found %d shards\n", len(shardFiles))

	// Prepare a result writer.
	var writer resultWriter

	// Process shards concurrently.
	var shardWg sync.WaitGroup
	shardCh := make(chan string, len(shardFiles))

	// Start a fixed number of shard workers (e.g. 40).
	numShardWorkers := 40
	for i := 0; i < numShardWorkers; i++ {
		go func() {
			for shardPath := range shardCh {
				processShard(shardPath, csvFiles, textColumns, outputDir, &writer)
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
	fmt.Printf("Processed all shards. Results have been appended per CSV (book) in: %s\n", outputDir)
}
